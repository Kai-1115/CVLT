import cv2
import numpy as np

def l2(img1, img2):
    diff = img1 - img2
    return np.sum(diff * diff)

def shift(img, dy, dx):
    shifted = np.roll(img, shift=dy, axis=0) 
    shifted = np.roll(shifted, shift=dx, axis=1) 
    return shifted

def l2_align(img1, img2): 
    best_score = float('inf')
    best_shift = (0, 0)

    for dy in range(-15, 16):
        for dx in range(-15, 16):
            shifted = shift(img2, dy, dx)
            score = l2(img1, shifted)

            if score < best_score:
                best_score = score
                best_shift = (dy, dx)

    return best_shift

if __name__ == '__main__':
    img = cv2.imread("cathedral.jpg", cv2.IMREAD_GRAYSCALE)

    h = img.shape[0] // 3
    b = img[0:h, :]
    g = img[h:2*h, :]
    r = img[2*h:3*h, :]

    # Normalize
    b = b.astype(np.float64) / 255.0
    g = g.astype(np.float64) / 255.0
    r = r.astype(np.float64) / 255.0
    
    l2_bg_shift = l2_align(b, g)
    l2_br_shift = l2_align(b, r)

    print("G -> B shift:", l2_bg_shift)
    print("R -> B shift:", l2_br_shift)

    g_aligned = shift(g, l2_bg_shift[0], l2_bg_shift[1])
    r_aligned = shift(r, l2_br_shift[0], l2_br_shift[1])

    l2_final = np.dstack((b, g_aligned, r_aligned)) # BGR

    output = (l2_final * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite("l2_final_cath.jpg", output)
