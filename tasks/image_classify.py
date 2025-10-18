from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

### ТУТ У МЕНЯ ПОЛУЧИЛОСЬ ПОЧИНИТЬ СЕРТИФИКАТЫ, ПОЭТОМУ РАБОТАЕМ НЕ ЛОКАЛЬНО)))

def main():
    project_root = Path(__file__).resolve().parent.parent
    img_path = project_root / "data" / "cat.jpg"
    if not img_path.exists():
        print(f"Нет файла: {img_path}/")
        return

    # устройство: MPS на M2, иначе CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("🧠 device:", device)

    # веса и предобработка из torchvision
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    # загрузка и преобразование
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1)[0]

    top5 = probs.topk(5)
    categories = [weights.meta["categories"][i] for i in top5.indices.tolist()]

    print("Топ-5 классов:")
    for cls, p in zip(categories, top5.values.tolist()):
        print(f"- {cls}: {p:.3f}")

if __name__ == "__main__":
    main()
