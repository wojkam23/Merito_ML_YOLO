#  Detekcja Logotypów Kanałów Telewizyjnych z wykorzystaniem YOLOv8

##  Cel projektu

Celem projektu jest stworzenie systemu, który automatycznie wykrywa logotypy kanałów telewizyjnych w materiałach wideo. Może to znaleźć zastosowanie w analizie treści nadawczych, wykrywaniu reklam, a także jako część systemów archiwizujących transmisje telewizyjne.

System bazuje na modelu detekcyjnym YOLOv8 (ang. *You Only Look Once*) oraz bibliotece OpenCV i umożliwia:

- jednoczesne przetwarzanie wielu strumieni wideo,
- detekcję logotypów stacji TV w czasie rzeczywistym,
- wizualizację wyników w formie siatki podglądowej.

---

##  Wykorzystane technologie

- **YOLOv8** – zaawansowany model do detekcji obiektów w czasie rzeczywistym (Ultralytics).
- **OpenCV** – biblioteka do przetwarzania obrazów i obsługi plików wideo.
- **Matplotlib** – do wizualizacji wyników w graficznym interfejsie.
- **Python 3.8+** – język programowania całego systemu.
- **Multithreading** – umożliwia równoległą analizę wielu kanałów.

---

##  Struktura projektu
├── logo_detection.py # Główna aplikacja – wykrywanie logotypów w czasie rzeczywistym

├── train_yolo.py # Skrypt treningowy YOLOv8 z dostosowanymi augmentacjami

├── rename_images.py # Zmiana nazw obrazów do spójnego formatu

├── crop_logos.py # Wycinanie logotypów z przykładowych obrazów (do podglądu)

├── split_val_set.py # Podział danych na zbiór treningowy i walidacyjny

├── sources.json # Mapowanie kanałów na ścieżki do plików wideo testowych

├── video_test/ # Przykładowe testowe pliki wideo

├── logos_preview/ # Folder z wyciętymi logotypami

---

##  Opis działania każdego komponentu

### `logo_detection.py`
Główna aplikacja – uruchamia wątki przetwarzające poszczególne kanały TV i co sekundę analizuje klatki wideo. Wykrywa logotypy przy pomocy modelu YOLOv8 i prezentuje wyniki w czasie rzeczywistym w formie siatki w oknie graficznym (`matplotlib`).

- Wątki analizują kanały równolegle.
- Każdy kanał otrzymuje swój podgląd i wykryty label (lub "REKLAMA").
- Detekcja oparta na wcześniej wytrenowanym modelu.

### `train_yolo.py`
Skrócony, zoptymalizowany skrypt do trenowania modelu YOLOv8. Używa `yolov8n.pt` jako bazy i wyłącza niepotrzebne augmentacje, takie jak zmiany kolorów czy odbicia, które nie mają zastosowania w przypadku logotypów.

### `rename_images.py`
Zmienia nazwy wszystkich obrazów w folderach treningowych na jednolity format: `frameTvn_000001.jpg`. Jest to ważne przy trenowaniu modelu i zachowaniu porządku w danych.

### `crop_logos.py`
Wycina logotypy z wcześniej oznaczonych obrazów (na podstawie współrzędnych YOLO). Użyteczne np. do szybkiego podglądu jakości danych i dla testów wizualnych.

### `split_val_set.py`
Losowo wybiera 20% obrazów z każdego kanału i przenosi je do zbioru walidacyjnego wraz z odpowiadającymi etykietami `.txt`. Zapewnia poprawny podział danych do treningu YOLO.

### `sources.json`
Plik konfiguracyjny zawierający mapowanie kanałów (np. `"tvn"`, `"ttv"`) na pliki testowe `.mp4`. Kanały bez przypisanej ścieżki są pomijane.

---

## Przykładowe dane

- Pliki wideo powinny znajdować się w folderze `video_test/` i być przypisane w `sources.json`.
- Obrazy treningowe i etykiety muszą być w formacie YOLO (center_x, center_y, width, height w skali 0–1).
- Detekcje są aktualizowane co sekundę (co 1 klatkę na sekundę), aby uniknąć przeciążenia systemu.

└── datasets/

└── tvchannels/

├── images/train/ # Obrazy treningowe

├── labels/train/ # Adnotacje YOLO

├── images/val/ # Obrazy walidacyjne

└── labels/val/ # Adnotacje walidacyjne

