from transformers import pipeline
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent
    texts_path = project_root / "data" / "texts.txt"
    if not texts_path.exists():
        print(f"NO {texts_path}")
        return

    clf = pipeline("sentiment-analysis")
    lines = [l.strip() for l in texts_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    print("Анализ тональности:")
    for t in lines:
        res = clf(t)[0]
        print(f"- {t}\n  → {res['label']} (score={res['score']:.3f})")

if __name__ == "__main__":
    main()
