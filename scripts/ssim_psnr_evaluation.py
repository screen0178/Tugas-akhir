import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load images
original_image = cv2.imread('original.png', cv2.IMREAD_GRAYSCALE)
reconstructed_image = cv2.imread('reconstructed.png', cv2.IMREAD_GRAYSCALE)

# Calculate PSNR
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

psnr_value = calculate_psnr(original_image, reconstructed_image)
print(f"PSNR: {psnr_value} dB")

# Calculate SSIMz
ssim_value, _ = ssim(original_image, reconstructed_image, full=True)
print(f"SSIM: {ssim_value}")