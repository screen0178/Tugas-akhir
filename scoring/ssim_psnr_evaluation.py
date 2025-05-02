import cv2
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Daftar nama file (tanpa ekstensi .png)
file_list = [
        "object_1_2x",
        "object_1_4x",
        "object_1_8x",
        "object_2_2x",
        "object_2_4x",
        "object_2_8x",
        "object_3_2x",
        "object_3_4x",
        "object_3_8x",
        "object_14_2x",
        "object_14_4x",
        "object_14_8x",
        "object_29_2x",
        "object_29_4x",
        "object_29_8x",
        "object_30_2x",
        "object_30_4x",
        "object_30_8x",
        "object_33_2x",
        "object_33_4x",
        "object_33_8x",
    ]

# Path folder gambar
folder_path = ""  # Ubah ke lokasi folder kamu

# Inisialisasi list hasil
results = []

def get_object_name(full_string):
    parts = full_string.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    else:
        return full_string

for name in file_list:
    # Buat path file
    sr_path = f"{folder_path}{get_object_name(name)}.jpg"        # SR file (hasil model)
    hr_path = f"{folder_path}{name}_out.jpg"     # HR file (ground truth)

    # Load gambar
    sr = cv2.imread(sr_path)
    hr = cv2.imread(hr_path)

    if sr is None or hr is None:
        print(f"File {name} tidak ditemukan, skip...")
        continue

    # Pastikan ukurannya sama
    if sr.shape != hr.shape:
        sr = cv2.resize(sr, (hr.shape[1], hr.shape[0]))

    # Hitung PSNR dan SSIM
    psnr_score = psnr(hr, sr, data_range=255)
    ssim_score = ssim(hr, sr, channel_axis=-1, data_range=255)

    # Simpan ke list
    results.append({
        "File Name": name,
        "PSNR (dB)": round(psnr_score, 2),
        "SSIM": round(ssim_score, 4)
    })

# Convert ke DataFrame
df = pd.DataFrame(results)

# Tampilkan hasil
print(df)

# Simpan ke CSV (opsional)
df.to_csv(f"{folder_path}evaluation_results.csv", index=False)



'''
object_1.jpg
PSNR: 26.23 dB
SSIM: 0.9389
 
object_2.jpg
PSNR: 22.86 dB
SSIM: 0.9182
 
object_3.jpg
PSNR: 27.18 dB
SSIM: 0.9334
 
object_14.jpg
PSNR: 27.84 dB
SSIM: 0.9303
 
object_29.jpg
PSNR: 26.67 dB
SSIM: 0.9420
 
object_30.jpg
PSNR: 28.01 dB
SSIM: 0.9556
 
object_33.jpg
PSNR: 26.23 dB
SSIM: 0.9389
 
'''