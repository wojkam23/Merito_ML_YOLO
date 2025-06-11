import os

# Główny folder zawierający podfoldery z obrazami każdego kanału TV
root_dir = "datasets/tvchannels/images/train"

# Pobranie listy nazw kanałów (czyli nazw podfolderów)
channels = sorted(os.listdir(root_dir))

# Iteracja po każdym kanale
for channel in channels:
    channel_path = os.path.join(root_dir, channel)

    # Sprawdzenie, czy ścieżka faktycznie jest folderem
    if not os.path.isdir(channel_path):
        continue

    # Pobranie wszystkich plików graficznych w folderze kanału (jpg/png/jpeg)
    files = sorted([
        f for f in os.listdir(channel_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    # Iteracja po plikach i zmiana nazw
    for idx, old_name in enumerate(files):
        ext = os.path.splitext(old_name)[1]  # Pobierz oryginalne rozszerzenie (np. .jpg)

        # Nowa nazwa w formacie: frame[ChannelName]_[numer_klatki]
        # Przykład: frameTvn_000001.jpg
        new_name = f"frame{channel.capitalize()}_{idx + 1:06d}{ext}"

        # Pełne ścieżki do pliku przed i po zmianie
        old_path = os.path.join(channel_path, old_name)
        new_path = os.path.join(channel_path, new_name)

        # Zmiana nazwy pliku
        os.rename(old_path, new_path)
        print(f"{old_name} → {new_name}")
