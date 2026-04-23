import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from skimage import io, color

def load_gray(path): # 轉灰階
    img = io.imread(path)
    if img.ndim == 3:
        img = img[:, :, :3]
        img = color.rgb2gray(img)
    return img.astype(float)

def load_rgb(path): # 轉彩色
    img = io.imread(path).astype(float)
    if img.max() > 1.0:
        img /= 255.0
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def save(fig, name):
    fig.savefig(name, dpi=150, bbox_inches='tight')
    print(f"Saved {name}")
    plt.close(fig)

def make_gaussian_kernel(ksize=15, sigma=2): # sigma 大 模糊強
    k1d = cv2.getGaussianKernel(ksize, sigma)
    return (k1d @ k1d.T)

def part1_1(img_path='cameraman.png'):
    img = load_gray(img_path)

    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])

    gx = convolve2d(img, dx, mode='same', boundary='symm')
    gy = convolve2d(img, dy, mode='same', boundary='symm')
    g_mag = np.sqrt(gx**2 + gy**2)

    threshold = 0.15
    edge_img = (g_mag > threshold).astype(float)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, im, title in zip(axes,
            [img, gx, gy, g_mag, edge_img],
            ['Original', 'Grad X', 'Grad Y', 'Grad Magnitude', f'Edges (t={threshold})']):
        ax.imshow(im, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    save(fig, 'part1_1_15.png')


def part1_2(img_path='cameraman.png'):
    img = load_gray(img_path)

    D_x = np.array([[1, -1]])
    D_y = np.array([[1], [-1]])
    G = make_gaussian_kernel(ksize=15, sigma=2)

    # Gaussian Blur
    blurred = convolve2d(img, G, mode='same', boundary='symm')
    gx1 = convolve2d(blurred, D_x, mode='same', boundary='symm')
    gy1 = convolve2d(blurred, D_y, mode='same', boundary='symm')
    mag1 = np.sqrt(gx1**2 + gy1**2)
    edge1 = (mag1 > 0.1).astype(float)

    # DoG
    DoG_x = convolve2d(G, D_x, mode='same', boundary='symm')
    DoG_y = convolve2d(G, D_y, mode='same', boundary='symm')
    gx2 = convolve2d(img, DoG_x, mode='same', boundary='symm')
    gy2 = convolve2d(img, DoG_y, mode='same', boundary='symm')
    mag2 = np.sqrt(gx2**2 + gy2**2)
    edge2 = (mag2 > 0.1).astype(float)

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    row1 = [img, blurred, gx1, gy1, edge1]
    row2 = [DoG_x, DoG_y, gx2, gy2, edge2]
    titles1 = ['Original', 'Blurred', 'Grad X (blur first)', 'Grad Y (blur first)', 'Edges (blur first)']
    titles2 = ['DoG X filter', 'DoG Y filter', 'Grad X (DoG)', 'Grad Y (DoG)', 'Edges (DoG)']
    for ax, im, t in zip(axes[0], row1, titles1):
        ax.imshow(im, cmap='gray'); ax.set_title(t); ax.axis('off')
    for ax, im, t in zip(axes[1], row2, titles2):
        ax.imshow(im, cmap='gray'); ax.set_title(t); ax.axis('off')
    plt.tight_layout()
    save(fig, 'part1_2_4.png')

# ─────────────────────────────────────────
# Part 2.1: Unsharp Masking
# ─────────────────────────────────────────

def sharpen(img, ksize=15, sigma=0.1, alpha=10):
    G = make_gaussian_kernel(ksize, sigma)
    if img.ndim == 3:
        blurred = np.stack([convolve2d(img[:,:,c], G, mode='same', boundary='symm') for c in range(3)], axis=2)
    else:
        blurred = convolve2d(img, G, mode='same', boundary='symm')
    return np.clip(img + alpha * (img - blurred), 0, 1)

def part2_1(blurry_path='butt.jpg', sharp_path=None):
    img = load_rgb(blurry_path)
    sharpened = sharpen(img)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.clip(img,  0,1)); axes[0].set_title('Original / Blurry'); axes[0].axis('off')
    axes[1].imshow(np.clip(sharpened,0,1)); axes[1].set_title('Sharpened');      axes[1].axis('off')
    plt.tight_layout()
    save(fig, 'part2_1_butt_1_10.png')

    # Blur a sharp image, then re-sharpen
    if sharp_path:
        orig = load_rgb(sharp_path)
        G = make_gaussian_kernel(21, 5)
        blurred = np.stack([convolve2d(orig[:,:,c], G, mode='same', boundary='symm')
                            for c in range(3)], axis=2)
        resharpened = sharpen(blurred)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, im, t in zip(axes,
                [orig, blurred, resharpened],
                ['Original Sharp', 'Blurred', 'Re-sharpened']):
            ax.imshow(np.clip(im,0,1)); ax.set_title(t); ax.axis('off')
        plt.tight_layout()
        save(fig, 'part2_1_resharpened_1_10.png')

# ─────────────────────────────────────────
# Part 2.2: Hybrid Images
# ─────────────────────────────────────────

def low_pass(img, ksize, sigma):
    G = make_gaussian_kernel(ksize, sigma)
    if img.ndim == 3:
        return np.stack([convolve2d(img[:,:,c], G, mode='same', boundary='symm')
                         for c in range(img.shape[2])], axis=2)
    return convolve2d(img, G, mode='same', boundary='symm')

def high_pass(img, ksize, sigma):
    return img - low_pass(img, ksize, sigma)

def hybrid_image(img1, img2, sigma_low=5, sigma_high=15, ksize=31):
    lo = low_pass(img1, ksize, sigma_low)
    hi = high_pass(img2, ksize, sigma_high)
    hybrid = np.clip(lo + hi, 0, 1)
    return lo, hi, hybrid

def show_fft(img_gray, ax, title):
    fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_gray))) + 1e-8)
    ax.imshow(fft, cmap='gray'); ax.set_title(title); ax.axis('off')

def part2_2(path1='derek.jpg', path2='nutmeg.jpg',
            sigma_low=5, sigma_high=15, ksize=31, use_color=True):
    img1 = load_rgb(path1)
    img2 = load_rgb(path2)

    # Resize img2 to match img1
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    lo, hi, hybrid = hybrid_image(img1, img2, sigma_low, sigma_high, ksize)

    # Result
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(img1, 0,1));   axes[0].set_title('Image 1 (low freq)');  axes[0].axis('off')
    axes[1].imshow(np.clip(img2, 0,1));   axes[1].set_title('Image 2 (high freq)'); axes[1].axis('off')
    axes[2].imshow(np.clip(hybrid,0,1));  axes[2].set_title('Hybrid');              axes[2].axis('off')
    plt.tight_layout()
    save(fig, 'part2_2_hybrid.png')

    # Fourier analysis (grayscale)
    g1     = color.rgb2gray(img1)
    g2     = color.rgb2gray(img2)
    g_lo   = color.rgb2gray(lo)
    g_hi   = np.clip(color.rgb2gray(hi) + 0.5, 0, 1)   # shift for visibility
    g_hyb  = color.rgb2gray(hybrid)

    fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    for ax, im, t in zip(axes[0],
            [img1, img2, lo, np.clip(hi+0.5,0,1), hybrid],
            ['Img1', 'Img2', 'Low-pass', 'High-pass', 'Hybrid']):
        ax.imshow(np.clip(im,0,1) if im.ndim==3 else np.clip(im,0,1), cmap='gray' if im.ndim==2 else None)
        ax.set_title(t); ax.axis('off')
    for ax, gim, t in zip(axes[1],
            [g1, g2, g_lo, g_hi, g_hyb],
            ['FFT Img1', 'FFT Img2', 'FFT Low-pass', 'FFT High-pass', 'FFT Hybrid']):
        show_fft(gim, ax, t)
    plt.tight_layout()
    save(fig, 'part2_2_fft.png')

# ─────────────────────────────────────────
# Part 2.3: Gaussian & Laplacian Stacks
# ─────────────────────────────────────────

def gaussian_stack(img, levels=6, ksize=31, sigma=2):
    stack = [img]
    for _ in range(levels - 1):
        stack.append(low_pass(stack[-1], ksize, sigma))
    return stack   # [original, blur1, blur2, ...]

def laplacian_stack(img, levels=6, ksize=31, sigma=2):
    g_stack = gaussian_stack(img, levels, ksize, sigma)
    l_stack = []
    for i in range(levels - 1):
        l_stack.append(g_stack[i] - g_stack[i+1])
    l_stack.append(g_stack[-1])   # last level = residual
    return l_stack

def part2_3(apple_path='apple.jpeg', orange_path='orange.jpeg'):
    apple  = load_rgb(apple_path)
    orange = load_rgb(orange_path)

    levels = 6
    ksize, sigma = 31, 4

    apple_lap  = laplacian_stack(apple,  levels, ksize, sigma)
    orange_lap = laplacian_stack(orange, levels, ksize, sigma)

    fig, axes = plt.subplots(2, levels, figsize=(24, 8))
    for i in range(levels):
        # Normalize each level for display
        al = apple_lap[i];  al = (al - al.min()) / (al.max() - al.min() + 1e-8)
        ol = orange_lap[i]; ol = (ol - ol.min()) / (ol.max() - ol.min() + 1e-8)
        axes[0, i].imshow(al); axes[0, i].set_title(f'Apple L{i}');  axes[0, i].axis('off')
        axes[1, i].imshow(ol); axes[1, i].set_title(f'Orange L{i}'); axes[1, i].axis('off')
    plt.tight_layout()
    save(fig, 'part2_3_stacks.png')

# ─────────────────────────────────────────
# Part 2.4: Multiresolution Blending
# ─────────────────────────────────────────

def blend(img1, img2, mask, levels=6, ksize=31, sigma=4):
    """
    img1, img2: RGB float [0,1]
    mask: float [0,1], same H×W (1 = img1, 0 = img2)
    """
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    l1 = laplacian_stack(img1, levels, ksize, sigma)
    l2 = laplacian_stack(img2, levels, ksize, sigma)
    gm = gaussian_stack(mask, levels, ksize, sigma)

    blended_stack = []
    for la, lb, gk in zip(l1, l2, gm):
        blended_stack.append(gk * la + (1 - gk) * lb)

    # Reconstruct by summing laplacian stack
    result = np.zeros_like(img1)
    for layer in blended_stack:
        result += layer
    return np.clip(result, 0, 1)

def vertical_mask(shape):
    """Left half = 1, right half = 0"""
    mask = np.zeros(shape[:2])
    mask[:, :shape[1]//2] = 1
    return mask

def part2_4_oraple(apple_path='apple.jpeg', orange_path='orange.jpeg'):
    apple  = load_rgb(apple_path)
    orange = load_rgb(orange_path)
    orange = cv2.resize(orange, (apple.shape[1], apple.shape[0]))

    mask   = vertical_mask(apple.shape)
    result = blend(apple, orange, mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(apple);                axes[0].set_title('Apple');  axes[0].axis('off')
    axes[1].imshow(orange);               axes[1].set_title('Orange'); axes[1].axis('off')
    axes[2].imshow(np.clip(result,0,1));  axes[2].set_title('Oraple'); axes[2].axis('off')
    plt.tight_layout()
    save(fig, 'part2_4_oraple.png')

    # Show blended Laplacian stack visualization (Figure 10 style)
    mask3 = mask[:, :, np.newaxis]
    l1 = laplacian_stack(apple,  6, 31, 4)
    l2 = laplacian_stack(orange, 6, 31, 4)
    gm = gaussian_stack(mask3,   6, 31, 4)

    fig, axes = plt.subplots(3, 6, figsize=(24, 10))
    for i in range(6):
        def norm(x): return np.clip((x - x.min())/(x.max()-x.min()+1e-8), 0, 1)
        axes[0,i].imshow(norm(l1[i]));               axes[0,i].set_title(f'Apple L{i}');   axes[0,i].axis('off')
        axes[1,i].imshow(norm(l2[i]));               axes[1,i].set_title(f'Orange L{i}');  axes[1,i].axis('off')
        axes[2,i].imshow(norm(gm[i]*l1[i]+(1-gm[i])*l2[i])  ); axes[2,i].set_title(f'Blend L{i}'); axes[2,i].axis('off')
    plt.tight_layout()
    save(fig, 'part2_4_laplacian_vis.png')

def part2_4_irregular(img1_path, img2_path, mask_path):
    """
    Blend with an irregular mask (white = img1, black = img2).
    Create your mask in any image editor and save as a grayscale PNG.
    """
    img1 = load_rgb(img1_path)
    img2 = load_rgb(img2_path)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    mask = io.imread(mask_path)
    if mask.ndim == 3:
        mask = color.rgb2gray(mask)
    mask = mask.astype(float)
    if mask.max() > 1.0:
        mask /= 255.0
    mask = cv2.resize(mask, (img1.shape[1], img1.shape[0]))

    result = blend(img1, img2, mask)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img1);               axes[0].set_title('Image 1'); axes[0].axis('off')
    axes[1].imshow(img2);               axes[1].set_title('Image 2'); axes[1].axis('off')
    axes[2].imshow(mask, cmap='gray');  axes[2].set_title('Mask');    axes[2].axis('off')
    axes[3].imshow(np.clip(result,0,1));axes[3].set_title('Blended'); axes[3].axis('off')
    plt.tight_layout()
    save(fig, 'part2_4_irregular.png')

# ─────────────────────────────────────────
# Run everything
# ─────────────────────────────────────────

if __name__ == '__main__':
    # Part 1
    part1_1('cameraman.png')
    part1_2('cameraman.png')

    # Part 2.1 – replace with your own sharp image path for the second argument
    part2_1('butt.jpg', sharp_path='butt.jpg')

    # Part 2.2 – replace with your own image pairs
    part2_2('dog.png', 'bmw.png', sigma_low=5, sigma_high=15, ksize=31)

    # Part 2.3
    part2_3('apple.jpeg', 'orange.jpeg')

    # Part 2.4 – vertical seam (oraple)
    part2_4_oraple('apple.jpeg', 'orange.jpeg')

    # Part 2.4 – irregular mask (supply your own images + mask)
    # part2_4_irregular('img1.jpg', 'img2.jpg', 'mask.png')