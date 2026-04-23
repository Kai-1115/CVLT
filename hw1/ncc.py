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

def ncc_align(img1, img2):
    best_score = float('-inf')
    best_shift = (0, 0)

    for dy in range(-15, 16):
        for dx in range(-15, 16):
            shifted = shift(img2, dy, dx)
            
            score = ncc(img1, shifted)

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

    ncc_bg_shift = ncc_align(b, g)
    ncc_br_shift = ncc_align(b, r)

    print("G -> B shift:", ncc_bg_shift)
    print("R -> B shift:", ncc_br_shift)

    g_aligned = shift(g, ncc_bg_shift[0], ncc_bg_shift[1])
    r_aligned = shift(r, ncc_br_shift[0], ncc_br_shift[1])

    ncc_final = np.dstack((b, g_aligned, r_aligned))  # BGR

    output = (ncc_final * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite("ncc_final_tobo.jpg", output)