import cv2
import json
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Backend GUI wymagany do interaktywnego wyświetlania matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO

#  Wczytanie ścieżek do plików wideo z pliku JSON
with open("sources.json", "r") as f:
    sources = json.load(f)

#  Lista kanałów (klasy do detekcji)
class_list = [
    "food", "metro", "travel", "ttv", "tvn", "tvn7", "tvn24",
    "tvnfabula", "tvnstyle", "tvnturbo", "warner"
]

#  Wczytanie wytrenowanego modelu YOLOv8 i jego optymalizacja
model = YOLO("runs/detect/train/weights/best.pt")
model.fuse()  # Łączenie warstw Convolution + BatchNorm dla szybszego działania

#  Parametry siatki do wizualizacji (layout 4x3)
cols = 4
rows = 3
cell_w = 320
cell_h = 180

#  Inicjalizacja struktur do przechowywania wyników
detections = {name: "Brak danych" for name in class_list}  # Wyniki detekcji
frames = {name: np.zeros((cell_h, cell_w, 3), dtype=np.uint8) for name in class_list}  # Ostatnia klatka z danego kanału

#  Funkcja analizująca pojedynczy stream wideo
def analyze_stream(channel, path):
    if not path:
        return  # Pominięcie pustych ścieżek

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps)  # Przetwarzanie 1 klatki na sekundę
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Koniec pliku wideo

        i += 1
        if i % skip_frames != 0:
            continue  # Pomijanie zbędnych klatek

        # 🔍 Detekcja obiektów przy pomocy YOLO
        results = model.predict(source=frame, conf=0.5, verbose=False)[0]
        cls = results.boxes.cls.tolist() if results.boxes else []  # Lista wykrytych klas
        names = model.names

        #  Przypisanie etykiety (pierwsza wykryta klasa) lub "REKLAMA", jeśli nic nie znaleziono
        label = names[int(cls[0])] if cls else "REKLAMA"
        detections[channel] = label

        #  Skalowanie klatki do rozmiaru siatki
        resized = cv2.resize(frame, (cell_w, cell_h))
        frames[channel] = resized

    cap.release()

#  Uruchomienie jednego wątku na każdy kanał (równoległe przetwarzanie)
threads = []
for channel in class_list:
    t = threading.Thread(target=analyze_stream, args=(channel, sources.get(channel, "")))
    t.start()
    threads.append(t)

# Inicjalizacja okna matplotlib do wyświetlania wyników
fig, ax = plt.subplots()
plt.ion()  # Tryb interaktywny (odświeżanie w pętli)

#  Główna pętla do wizualizacji w czasie rzeczywistym
while True:
    # Pusta "mozaika" (siatka z kanałami)
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for idx, name in enumerate(class_list):
        row = idx // cols
        col = idx % cols
        x1, y1 = col * cell_w, row * cell_h

        frame = frames[name]
        label = detections[name]

        #  Dodanie podpisu z nazwą kanału i aktualnym wynikiem detekcji
        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"{name.upper()}: {label}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        # Umieszczenie klatki w odpowiednim miejscu mozaiki
        canvas[y1:y1 + cell_h, x1:x1 + cell_w] = overlay

    # Aktualizacja wyświetlanego obrazu w matplotlib
    ax.clear()
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title("TP16 - Logo Detection")
    ax.axis("off")
    plt.pause(0.05)  # Odświeżenie co 50 ms (~20 FPS)
