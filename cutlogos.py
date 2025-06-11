import os
import cv2
import glob

# Słownik zawierający współrzędne logotypów (YOLOv8 format):
# [środek_x, środek_y, szerokość, wysokość] – wszystko w postaci znormalizowanej (0–1)
logos = {
    "food": (0.904688, 0.896759, 0.072917, 0.126852),
    "metro": (0.873438, 0.104167, 0.113542, 0.060185),
    "travel": (0.877083, 0.822222, 0.120833, 0.085185),
    "ttv": (0.100260, 0.113426, 0.056771, 0.087963),
    "tvn": (0.894792, 0.110185, 0.061458, 0.111111),
    "tvn7": (0.886719, 0.109722, 0.068229, 0.087963),
    "tvn24": (0.089844, 0.810185, 0.073438, 0.114815),
    "tvnfabula": (0.098958, 0.109722, 0.062500, 0.110185),
    "tvnstyle": (0.911719, 0.124537, 0.065104, 0.115741),
    "tvnturbo": (0.161719, 0.117130, 0.127604, 0.056481),
    "warner": (0.926823, 0.091204, 0.050521, 0.087963)
}

# Ścieżka do folderu z obrazami poszczególnych kanałów
image_dir = "datasets/tvchannels/images/train"

# Folder do zapisu wyciętych logotypów
output_dir = "logos_preview"
os.makedirs(output_dir, exist_ok=True)  # Tworzy folder, jeśli nie istnieje

# Rozdzielczość obrazów (używana do przeliczenia współrzędnych z YOLO do pikseli)
image_w, image_h = 1920, 1080

# Iteracja po wszystkich kanałach
for channel, bbox in logos.items():
    # Szukaj obrazów danego kanału (po nazwie)
    pattern = os.path.join(image_dir, channel, f"frame_{channel}_*")
    images = glob.glob(pattern)

    if not images:
        print(f"⚠️ Brak obrazów dla kanału: {channel}")
        continue  # Jeśli brak obrazów – pomiń

    # Weź pierwszy obraz z folderu
    img_path = images[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Nie udało się wczytać: {img_path}")
        continue

    # Rozpakowanie współrzędnych YOLO
    x_c, y_c, w, h = bbox

    # Przeliczenie współrzędnych z normalizowanych na pikselowe
    x_center = int(x_c * image_w)
    y_center = int(y_c * image_h)
    width = int(w * image_w)
    height = int(h * image_h)

    # Obliczenie punktów narożnych prostokąta z zachowaniem granic obrazu
    x1 = max(0, x_center - width // 2)
    y1 = max(0, y_center - height // 2)
    x2 = min(image_w, x_center + width // 2)
    y2 = min(image_h, y_center + height // 2)

    # Wycięcie logotypu z obrazu
    cropped = img[y1:y2, x1:x2]

    # Zapis wyciętego logotypu do pliku
    out_path = os.path.join(output_dir, f"{channel}.jpg")
    cv2.imwrite(out_path, cropped)
    print(f"✅ Wycięto logo: {channel} -> {out_path}")
