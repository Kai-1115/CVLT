import cv2
import numpy as np

def ncc(img1, img2):
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)

    img1_m = img1 - mean1
    img2_m = img2 - mean2

    b = np.sum(img1_m * img2_m)
    a = np.sqrt(np.sum(img1_m ** 2) * np.sum(img2_m ** 2))

    if a == 0:
        return -1

    return b / a

def shift(img, dy, dx):
    shifted = np.roll(img, shift=dy, axis=0)
    shifted = np.roll(shifted, shift=dx, axis=1)
    return shifted

def crop_border(img, ratio=0.1):
    h, w = img.shape[:2]
    y1 = int(h * ratio)
    y2 = int(h * (1 - ratio))
    x1 = int(w * ratio)
    x2 = int(w * (1 - ratio))
    return img[y1:y2, x1:x2]

def pyramid_align(img1, img2, level):
    # terminal
    if level == 0:
        best_score = -1
        best_shift = (0, 0)

        ref = crop_border(img1, 0.1)

        for dy in range(-15, 16):
            for dx in range(-15, 16):
                shifted = shift(img2, dy, dx)
                shifted_crop = crop_border(shifted, 0.1)
                score = ncc(ref, shifted_crop)

                if score > best_score:
                    best_score = score
                    best_shift = (dy, dx)

        return best_shift

    # 縮小圖片
    small_img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2))
    small_img2 = cv2.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2))

    small_shift = pyramid_align(small_img1, small_img2, level - 1)

    base_dy = small_shift[0] * 2
    base_dx = small_shift[1] * 2

    best_score = -1
    best_shift = (base_dy, base_dx)

    ref = crop_border(img1, 0.1)

    for dy in range(base_dy - 2, base_dy + 3):
        for dx in range(base_dx - 2, base_dx + 3):
            shifted = shift(img2, dy, dx)
            shifted_crop = crop_border(shifted, 0.1)
            score = ncc(ref, shifted_crop)

            if score > best_score:
                best_score = score
                best_shift = (dy, dx)

    return best_shift

if __name__ == '__main__':
    img = cv2.imread("tobolsk.jpg", cv2.IMREAD_GRAYSCALE)

    h = img.shape[0] // 3
    b = img[0:h, :]
    g = img[h:2*h, :]
    r = img[2*h:3*h, :]

    # Normalize
    b = b.astype(np.float64) / 255.0
    g = g.astype(np.float64) / 255.0
    r = r.astype(np.float64) / 255.0

    # pyramid alignment
    level = 2
    bg_shift = pyramid_align(b, g, level)
    br_shift = pyramid_align(b, r, level)

    print("G -> B shift:", bg_shift)
    print("R -> B shift:", br_shift)

    g_aligned = shift(g, bg_shift[0], bg_shift[1])
    r_aligned = shift(r, br_shift[0], br_shift[1])

    final_img = np.dstack((b, g_aligned, r_aligned))   # BGR
    output = (final_img * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite("pyramid2_tobo.jpg", output)