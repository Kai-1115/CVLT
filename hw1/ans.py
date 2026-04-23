import os
import cv2
import numpy as np


# =========================
# Basic utilities
# =========================
def read_image(path):
    """
    Read image as float32 in range [0, 1].
    Supports grayscale jpg / tif.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    # If accidentally loaded as color, convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    # normalize according to dtype scale
    if img.max() > 1.0:
        img /= img.max()

    return img


def crop_to_three_parts(img):
    """
    The input glass plate image is stacked vertically in BGR order:
    top    -> B
    middle -> G
    bottom -> R
    """
    h = img.shape[0] // 3
    w = img.shape[1]

    B = img[0:h, :]
    G = img[h:2*h, :]
    R = img[2*h:3*h, :]

    return B, G, R


def remove_border(img, border_ratio=0.1):
    """
    Crop away outer border when computing alignment metric.
    This helps reduce border artifacts.
    """
    h, w = img.shape
    dy = int(h * border_ratio)
    dx = int(w * border_ratio)
    return img[dy:h-dy, dx:w-dx]


def shift_image(img, dx, dy):
    """
    Shift image by (dx, dy) using np.roll.
    dx: horizontal shift
    dy: vertical shift
    """
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


# =========================
# Matching metrics
# =========================
def ncc_score(img1, img2):
    """
    Normalized Cross Correlation.
    Higher is better.
    """
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)

    a = a - np.mean(a)
    b = b - np.mean(b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-8 or norm_b < 1e-8:
        return -1.0

    return np.sum(a * b) / (norm_a * norm_b)


def edge_map(img):
    """
    Use Sobel gradient magnitude as feature for more robust alignment.
    Often works better than raw intensity when color channels differ.
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag


# =========================
# Single-scale exhaustive search
# =========================
def align_single_scale(ref, target, search_range=15, use_edges=True):
    """
    Align target to ref using exhaustive search in [-search_range, search_range].
    Returns best (dx, dy).
    """
    if use_edges:
        ref_feature = edge_map(ref)
        target_feature = edge_map(target)
    else:
        ref_feature = ref
        target_feature = target

    ref_feature = remove_border(ref_feature, border_ratio=0.1)
    best_score = -1e18
    best_dx, best_dy = 0, 0

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            shifted = shift_image(target_feature, dx, dy)
            shifted = remove_border(shifted, border_ratio=0.1)

            score = ncc_score(ref_feature, shifted)

            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    return best_dx, best_dy


# =========================
# Pyramid alignment
# =========================
def build_pyramid(img, min_size=200):
    """
    Build image pyramid from original to coarse.
    pyramid[0] = original
    pyramid[-1] = coarsest
    """
    pyramid = [img]
    current = img

    while min(current.shape[0], current.shape[1]) > min_size:
        current = cv2.resize(
            current,
            (current.shape[1] // 2, current.shape[0] // 2),
            interpolation=cv2.INTER_AREA
        )
        pyramid.append(current)

    return pyramid


def align_pyramid(ref, target, use_edges=True):
    """
    Coarse-to-fine pyramid alignment.
    Returns best (dx, dy) shifting target to ref.
    """
    ref_pyr = build_pyramid(ref)
    tgt_pyr = build_pyramid(target)

    # Start from coarsest level
    level = len(ref_pyr) - 1
    dx, dy = 0, 0

    while level >= 0:
        ref_l = ref_pyr[level]
        tgt_l = tgt_pyr[level]

        # Shift target by current estimate
        shifted = shift_image(tgt_l, dx, dy)

        # Smaller search window per level
        local_dx, local_dy = align_single_scale(
            ref_l, shifted, search_range=4, use_edges=use_edges
        )

        dx += local_dx
        dy += local_dy

        # Scale shift for next finer level
        if level > 0:
            dx *= 2
            dy *= 2

        level -= 1

    return dx, dy


# =========================
# Auto crop final RGB image
# =========================
def auto_crop_color_image(rgb, threshold=0.05):
    """
    Simple automatic crop:
    remove rows/cols near borders where channels disagree too much or are too dark/bright.
    """
    rgb = np.clip(rgb, 0, 1)
    h, w, _ = rgb.shape

    # Channel disagreement map
    diff_rg = np.abs(rgb[:, :, 0] - rgb[:, :, 1])
    diff_rb = np.abs(rgb[:, :, 0] - rgb[:, :, 2])
    diff_gb = np.abs(rgb[:, :, 1] - rgb[:, :, 2])
    disagreement = (diff_rg + diff_rb + diff_gb) / 3.0

    row_score = np.mean(disagreement, axis=1)
    col_score = np.mean(disagreement, axis=0)

    top = 0
    while top < h // 2 and row_score[top] > threshold:
        top += 1

    bottom = h - 1
    while bottom > h // 2 and row_score[bottom] > threshold:
        bottom -= 1

    left = 0
    while left < w // 2 and col_score[left] > threshold:
        left += 1

    right = w - 1
    while right > w // 2 and col_score[right] > threshold:
        right -= 1

    # safety check
    if top >= bottom or left >= right:
        return rgb

    return rgb[top:bottom+1, left:right+1, :]


# =========================
# Main colorization function
# =========================
def colorize_glass_plate(img, use_pyramid=True, use_edges=True, auto_crop=True):
    """
    Input: vertically stacked grayscale plate
    Output:
      rgb image
      G shift relative to B
      R shift relative to B
    """
    B, G, R = crop_to_three_parts(img)

    if use_pyramid:
        g_dx, g_dy = align_pyramid(B, G, use_edges=use_edges)
        r_dx, r_dy = align_pyramid(B, R, use_edges=use_edges)
    else:
        g_dx, g_dy = align_single_scale(B, G, search_range=15, use_edges=use_edges)
        r_dx, r_dy = align_single_scale(B, R, search_range=15, use_edges=use_edges)

    G_aligned = shift_image(G, g_dx, g_dy)
    R_aligned = shift_image(R, r_dx, r_dy)

    # Stack into RGB
    rgb = np.dstack([R_aligned, G_aligned, B])

    # Optional crop
    if auto_crop:
        rgb = auto_crop_color_image(rgb)

    rgb = np.clip(rgb, 0, 1)
    return rgb, (g_dx, g_dy), (r_dx, r_dy)


# =========================
# Save utility
# =========================
def save_rgb_image(path, rgb):
    """
    Save RGB float image to disk as uint8.
    """
    out = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out_bgr)


# =========================
# Example main
# =========================
def process_one_image(input_path, output_dir="results",
                      use_pyramid=True, use_edges=True, auto_crop=True):
    os.makedirs(output_dir, exist_ok=True)

    img = read_image(input_path)
    rgb, g_shift, r_shift = colorize_glass_plate(
        img,
        use_pyramid=use_pyramid,
        use_edges=use_edges,
        auto_crop=auto_crop
    )

    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_colorized.jpg")
    save_rgb_image(output_path, rgb)

    print(f"Image: {input_path}")
    print(f"G aligned to B: dx={g_shift[0]}, dy={g_shift[1]}")
    print(f"R aligned to B: dx={r_shift[0]}, dy={r_shift[1]}")
    print(f"Saved to: {output_path}")


def process_folder(input_dir="data", output_dir="results",
                   use_pyramid=True, use_edges=True, auto_crop=True):
    os.makedirs(output_dir, exist_ok=True)

    valid_ext = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

    for name in os.listdir(input_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in valid_ext:
            path = os.path.join(input_dir, name)
            try:
                process_one_image(
                    path,
                    output_dir=output_dir,
                    use_pyramid=use_pyramid,
                    use_edges=use_edges,
                    auto_crop=auto_crop
                )
                print("-" * 50)
            except Exception as e:
                print(f"Failed on {path}: {e}")


if __name__ == "__main__":
    # Example 1: process one image
    # process_one_image("data/cathedral.jpg")

    # Example 2: process all images in a folder
    process_folder("data", "results", use_pyramid=True, use_edges=True, auto_crop=True)