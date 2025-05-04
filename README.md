# UTS-VISI-KOMPUTER-

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import os  # Untuk pengecekan path file

def hitung_mse(citra_asli, citra_hasil):
  """Menghitung Mean Squared Error (MSE) antara dua citra."""
  mse = np.mean((citra_asli - citra_hasil) ** 2)
  return mse

def hitung_rmse(mse):
  """Menghitung Root Mean Squared Error (RMSE) dari MSE."""
  rmse = np.sqrt(mse)
  return rmse

def hitung_psnr(citra_asli, citra_hasil):
  """Menghitung Peak Signal-to-Noise Ratio (PSNR) antara dua citra."""
  mse = hitung_mse(citra_asli, citra_hasil)
  if mse == 0:
    return float('inf')
  psnr = 10 * np.log10((255 ** 2) / mse)
  return psnr

def morfologi(citra, kernel_size, metode):
  """Melakukan operasi morfologi pada citra."""
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  if metode == "dilasi":
    hasil = cv2.dilate(citra, kernel, iterations=1)
  elif metode == "erosi":
    hasil = cv2.erode(citra, kernel, iterations=1)
  elif metode == "closing":
    hasil = cv2.morphologyEx(citra, cv2.MORPH_CLOSE, kernel)
  elif metode == "opening":
    hasil = cv2.morphologyEx(citra, cv2.MORPH_OPEN, kernel)
  else:
    raise ValueError("Metode morfologi tidak valid.")
  return hasil

# Fungsi untuk mencoba membaca citra dengan penanganan error yang lebih baik
def coba_baca_citra(path_citra):
  """Mencoba membaca citra dan menangani error jika gagal."""
  if not os.path.exists(path_citra):
    print(f"Error: File citra tidak ditemukan di '{path_citra}'.")
    return None
  try:
    citra = cv2.imread(path_citra, cv2.IMREAD_GRAYSCALE)
    if citra is None:
      print(f"Error: Gagal membaca citra dari '{path_citra}'. Pastikan format file didukung.")
    return citra
  except Exception as e:
    print(f"Error: Terjadi kesalahan saat membaca citra: {e}")
    return None

# Load citra dengan fungsi coba_baca_citra()
path_citra = "download.jpg"  # Ganti dengan path yang benar!
citra_asli = coba_baca_citra(path_citra)

# Jika citra gagal dibaca, coba lagi atau keluar
while citra_asli is None:
  path_citra_baru = input("Masukkan path citra yang benar (atau ketik 'keluar'): ")
  if path_citra_baru.lower() == 'keluar':
    print("Program dihentikan.")
    exit()
  citra_asli = coba_baca_citra(path_citra_baru)

# Kernel sizes
kernel_sizes = [11, 13]
metode_morfologi = ["dilasi", "erosi", "closing", "opening"]

for kernel_size in kernel_sizes:
  print(f"\nHasil untuk Kernel {kernel_size}:")
  for metode in metode_morfologi:
    citra_hasil = morfologi(citra_asli.copy(), kernel_size, metode)
    mse = hitung_mse(citra_asli, citra_hasil)
    rmse = hitung_rmse(mse)
    psnr = hitung_psnr(citra_asli, citra_hasil)

    print(f"  Metode: {metode.capitalize()}")
    print(f"    MSE: {mse:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    PSNR: {psnr:.2f} dB")

    cv2_imshow(citra_hasil)
