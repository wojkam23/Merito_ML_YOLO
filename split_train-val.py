import os
import random
import shutil
from pathlib import Path

# Parametry konfiguracyjne
base_dir = "datasets/tvchannels"
image_exts = [".jpg", ".jpeg", ".png"]
split_ratio = 0.2  # 20% danych zostanie przeniesione do zbioru walidacyjnego

# Ścieżki do folderów obrazów i etykiet (train / val)
image_train_root = Path(base_dir) / "images" / "train"
label_train_root = Path(base_dir) / "labels" / "train"
image_val_root   = Path(base_dir) / "images" / "val"
label_val_root   = Path(base_dir) / "labels" / "val"

# Lista kanałów (czyli folderów) na podstawie folderu treningowego
channels = [d.name for d in image_train_root.iterdir() if d.is_dir()]

# Przetwarzanie każdego kanału osobno
for channel in channels:
    # Ścieżki źródłowe i docelowe dla obrazów i etykiet
    image_dir = image_train_root / channel
    label_dir = label_train_root / channel
    image_val_dir = image_val_root / channel
    label_val_dir = label_val_root / channel

    # Utworzenie folderów walidacyjnych, jeśli nie istnieją
    image_val_dir.mkdir(parents=True, exist_ok=True)
    label_val_dir.mkdir(parents=True, exist_ok=True)

    # Lista wszystkich plików graficznych
    all_images = [f for f in image_dir.iterdir() if f.suffix.lower() in image_exts]
    if not all_images:
        continue  # Pomija kanał bez obrazów

    # Losowy wybór obrazów do walidacji wg podanego stosunku
    val_images = random.sample(all_images, int(len(all_images) * split_ratio))

    # Przenoszenie obrazów i ich etykiet
    for img_path in val_images:
        # Przenieś obraz do folderu walidacyjnego
        target_img = image_val_dir / img_path.name
        shutil.move(str(img_path), str(target_img))

        # Przenieś odpowiadającą etykietę (plik .txt)
        label_name = img_path.with_suffix(".txt").name  # Zamiana rozszerzenia .jpg → .txt
        label_src = label_dir / label_name
        label_dst = label_val_dir / label_name

        if label_src.exists():
            shutil.move(str(label_src), str(label_dst))
        else:
            print(f" Brak etykiety dla {img_path.name}")

    print(f" {channel}: przeniesiono {len(val_images)} obrazów do walidacji")
