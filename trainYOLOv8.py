from ultralytics import YOLO

# Wczytanie pretrenowanego, lekkiego modelu YOLOv8n (nano)
# Można też użyć innych wariantów: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
model = YOLO("yolov8n.pt")

# Rozpoczęcie treningu modelu
model.train(
    data="data.yaml",     # Ścieżka do pliku konfiguracyjnego z klasami i danymi
    epochs=5,             # Liczba epok treningu
    imgsz=640,            # Rozmiar obrazów (YOLO przeskaluje do tego rozmiaru)
    batch=16,             # Wielkość batcha (liczba obrazów przetwarzanych na raz)
    optimizer="AdamW",    # Optymalizator – AdamW zamiast standardowego SGD

    # Wyłączone augmentacje kolorystyczne – logo ma stałe barwy
    hsv_h=0.0,            # Brak zmiany odcienia
    hsv_s=0.0,            # Brak zmiany nasycenia
    hsv_v=0.0,            # Brak zmiany jasności

    # Lekka augmentacja przestrzenna
    translate=0.05,       # Przesunięcie obrazu do 5%
    scale=0.1,            # Skalowanie (powiększenie/pomniejszenie) do 10%

    # Brak odbić lustrzanych – logotypy nigdy nie są lustrzane
    fliplr=0.0,           # Brak odbicia w poziomie
    flipud=0.0            # Brak odbicia w pionie
)
