import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
from pathlib import Path
import csv, librosa

### НЕ РАБОТАЕТ У МЕНЯ SSL ПРИШЛОСЬ ВСЁ ЛОКАЛЬНО ДЕЛАТЬ :D

def load_labels_local():
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "yamnet" / "assets" / "yamnet_class_map.csv"
    if not csv_path.exists():
        print(f"Нет файла: {csv_path}")
        return None
    labels = []
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            labels.append(row["display_name"])
    return labels

def load_yamnet_local():
    project_root = Path(__file__).resolve().parent.parent
    local_dir = project_root / "data" / "yamnet"
    if not local_dir.exists():
        print(f"Нет локальной модели: {local_dir}\n")
        return None
    return hub.load(str(local_dir))

def main():
    project_root = Path(__file__).resolve().parent.parent
    wav_path = project_root / "data" / "sample.wav"
    if not wav_path.exists():
        print(f"Нет файла: {wav_path}\nПоложи короткий WAV (3–10 сек, 16 kHz mono).")
        return

    labels = load_labels_local()
    if labels is None:
        return

    model = load_yamnet_local()
    if model is None:
        return

    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    scores, _, _ = model(audio)
    mean_scores = np.mean(scores.numpy(), axis=0)
    top5_idx = np.argsort(mean_scores)[-5:][::-1]

    print("Топ-5 распознанных звуков:")
    for i in top5_idx:
        print(f"- {labels[i]}: {mean_scores[i]:.3f}")

if __name__ == "__main__":
    main()
