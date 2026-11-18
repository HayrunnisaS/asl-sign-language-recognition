import kagglehub
import shutil
import os
import random

# 1. Veri setini Kaggle üzerinden indir
print("Veri seti indiriliyor, lütfen bekleyin (yaklaşık 1 GB)...")
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("İndirme tamamlandı. Ham yol:", path)

source_dir = os.path.join(path, "asl_alphabet_train", "asl_alphabet_train")
target_dir = "./dataset"

# 2. Sadece 26 harfi al ve her sınıftan dengeli (600 adet) görüntü kopyala
MAX_IMAGES_PER_CLASS = 600  # Eğitimi hızlı ve verimli tutmak için
valid_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A'dan Z'ye 26 harf

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    random.seed(42)

    for letter in valid_letters:
        src_letter_path = os.path.join(source_dir, letter)
        dst_letter_path = os.path.join(target_dir, letter)
        os.makedirs(dst_letter_path, exist_ok=True)

        all_images = os.listdir(src_letter_path)
        selected_images = random.sample(all_images, min(MAX_IMAGES_PER_CLASS, len(all_images)))

        for img_name in selected_images:
            shutil.copy(
                os.path.join(src_letter_path, img_name),
                os.path.join(dst_letter_path, img_name)
            )
        print(f"'{letter}' harfi hazırlandı ({len(selected_images)} görsel).")

    print("\nTüm 26 harflik veri seti './dataset' klasörüne başarıyla kuruldu!")
else:
    print("'./dataset' klasörü zaten mevcut.")