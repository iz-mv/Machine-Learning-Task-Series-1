from pathlib import Path
from ultralytics import YOLO

def main():
    project_root = Path(__file__).resolve().parent.parent
    video_in = project_root / "data" / "sample.mp4"
    if not video_in.exists():
        print(f"Нет видео: {video_in}")
        return

    outputs = project_root / "outputs"
    outputs.mkdir(exist_ok=True)

    #самая легкая модель
    model = YOLO("yolov8n.pt")

    result = model.predict(
        source=str(video_in),
        save=True, # сохранить
        project=str(outputs), # корневая папка результатов
        name="yolo", # подпапка
        vid_stride=1, # каждый кадр
        device="mps" # сам уйдет на cpu, если что
    )

    print("Готово!")
    print(f"Посмотри результат здесь: {outputs / 'yolo'}")

if __name__ == "__main__":
    main()
