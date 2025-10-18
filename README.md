# 🧠 Machine Learning Tasks / Задания по машинному обучению  
**SE 25/26 — Task Series 1**

---

## 🇬🇧 English Description

This project demonstrates the use of **ready-made AI / ML / DL models** to solve applied problems in **text, audio, image, and video processing**, as well as deploying a **local LLM (Large Language Model)** on a personal computer.

All tasks are implemented **locally** on macOS (Apple Silicon, M2), using **pre-trained models** without any retraining or fine-tuning.

### 📘 Project Goal
To explore the capabilities of modern AI frameworks and libraries (Hugging Face, TensorFlow, PyTorch, Ultralytics, llama.cpp) and apply them to real problems with minimal code.

---

### 📂 Project Structure
```text
api/
 ├── llm_api.py           # prototype of LLM API
 ├── test_llm_api.py      # API testing script
 └── llm_api_start.txt    # launch notes
data/
 ├── yamnet/              # audio labels
 ├── cat.jpg              # sample image
 ├── model.gguf           # Llama 3.2 3B Instruct (GGUF)
 ├── sample.wav           # sample audio
 ├── sample.mp4           # sample video
 └── texts.txt            # example texts
outputs/
 ├── yolo/                # YOLOv8 results
 ├── llm_chat_*.txt       # saved LLM chat sessions
tasks/
 ├── audio_yamnet.py      # sound classification (TensorFlow)
 ├── image_classify.py    # image classification (PyTorch)
 ├── llm_local.py         # single-prompt LLM test
 ├── text_sentiment.py    # text sentiment analysis (Hugging Face)
 ├── video_yolo.py        # object detection in video (YOLOv8)
 └── yolov8n.pt           # YOLOv8 weights
requirements.txt
README.md
REPORT.md
link_on_llama.txt
```

---

### ⚙️ Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you get SSL errors on macOS:
```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```

---

### 🚀 Running the tasks

#### 1️⃣ Text Sentiment (Hugging Face)
```bash
python tasks/text_sentiment.py
```
➡️ Detects text sentiment (positive/negative).

#### 2️⃣ Audio Classification (TensorFlow Hub / YAMNet)
```bash
python tasks/audio_yamnet.py
```
➡️ Recognizes sounds from a short WAV file.

#### 3️⃣ Image Classification (PyTorch / ResNet50)
```bash
python tasks/image_classify.py
```
➡️ Classifies an image (e.g., dog, cat, etc.).

#### 4️⃣ Object Detection in Video (YOLOv8)
```bash
python tasks/video_yolo.py
```
➡️ Detects objects frame-by-frame in a video.

#### 5️⃣ Local LLM Chat (Llama.cpp)
```bash
python tasks/llm_chat.py
```
➡️ Runs a local assistant completely offline.

Download model (for example):
> `Llama-3.2-3B-Instruct.Q4_K_M.gguf`  
and put it into:
> `data/model.gguf`

---

### 🧩 Frameworks Used

| Task Type | Library / Framework |
|------------|--------------------|
| Text | Hugging Face Transformers |
| Audio | TensorFlow Hub (YAMNet) |
| Image | PyTorch / torchvision |
| Video | Ultralytics YOLOv8 |
| LLM | Llama.cpp (llama-cpp-python) |

---

### 📈 Performance (MacBook Air M2, 8GB RAM)

| Task | Framework | RAM Usage | Runtime |
|------|------------|-----------|----------|
| Text | Transformers | ~0.5 GB | <1s |
| Audio | TensorFlow | ~1 GB | 2–3s |
| Image | PyTorch | ~1 GB | <2s |
| Video | YOLOv8n | ~2 GB | real-time (CPU/MPS) |
| LLM | llama.cpp (3B Q4_K_M) | ~5–6 GB | interactive |

---

### 👤 Author
**Islam Mubarakov**  
3rd year Software Engineering, Kazan Federal University  
Course: *SE 25/26 — Task Series 1*

---

## 🇷🇺 Описание на русском

Этот проект демонстрирует использование **готовых моделей машинного обучения (AI / ML / DL)**  
для решения прикладных задач: анализа текста, звука, изображений, видео,  
а также — развёртывание **локальной языковой модели (LLM)** прямо на компьютере (без интернета).

Все решения работают **полностью локально**, без дообучения,  
и используют **предобученные модели** из популярных библиотек.

---

### 🎯 Цель проекта
Познакомиться с возможностями современных библиотек  
(Hugging Face, TensorFlow, PyTorch, Ultralytics, llama.cpp)  
и применить их к реальным задачам с минимальным кодом.

---

### 📂 Структура проекта
```text
api/
 ├── llm_api.py           # прототип API для LLM
 ├── test_llm_api.py      # тестовый скрипт для API
 └── llm_api_start.txt    # заметки по запуску
data/
 ├── yamnet/              # метки аудио
 ├── cat.jpg              # пример изображения
 ├── model.gguf           # Llama 3.2 3B Instruct (GGUF)
 ├── sample.wav           # пример аудио
 ├── sample.mp4           # пример видео
 └── texts.txt            # примеры текстов
outputs/
 ├── yolo/                # результаты YOLOv8
 ├── llm_chat_*.txt       # сохранённые сессии чата LLM
tasks/
 ├── audio_yamnet.py      # классификация звуков (TensorFlow)
 ├── image_classify.py    # классификация изображений (PyTorch)
 ├── llm_local.py         # тест одного запроса к LLM
 ├── text_sentiment.py    # анализ тональности текста (Hugging Face)
 ├── video_yolo.py        # детекция объектов в видео (YOLOv8)
 └── yolov8n.pt           # веса модели YOLOv8
requirements.txt
README.md
REPORT.md
link_on_llama.txt
```

---

### ⚙️ Установка и запуск
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 🚀 Выполнение заданий

#### 1️⃣ Анализ текста (Transformers)
```bash
python tasks/text_sentiment.py
```
Определяет тональность (положительная / отрицательная).

#### 2️⃣ Классификация звука (TensorFlow / YAMNet)
```bash
python tasks/audio_yamnet.py
```
Определяет тип звука из короткого WAV-файла.

#### 3️⃣ Классификация изображений (PyTorch)
```bash
python tasks/image_classify.py
```
Определяет, что изображено на фото (собака, кошка и т.д.).

#### 4️⃣ Детекция объектов в видео (YOLOv8)
```bash
python tasks/video_yolo.py
```
Находит объекты на каждом кадре видео.

#### 5️⃣ Локальный чат с LLM (llama.cpp)
```bash
python tasks/llm_chat.py
```
Запускает чат с языковой моделью полностью офлайн.

> Для работы скачайте модель  
> **Llama-3.2-3B-Instruct.Q4_K_M.gguf**  
> и поместите в папку `data/model.gguf`

---

### 🧩 Используемые библиотеки
| Тип задачи | Библиотека |
|-------------|-------------|
| Текст | Hugging Face Transformers |
| Аудио | TensorFlow Hub (YAMNet) |
| Изображения | PyTorch / torchvision |
| Видео | Ultralytics YOLOv8 |
| LLM | Llama.cpp (llama-cpp-python) |

---

### 📊 Производительность (MacBook Air M2, 8GB RAM)
| Задача | Фреймворк | Память | Время |
|--------|------------|--------|--------|
| Текст | Transformers | ~0.5 ГБ | <1с |
| Аудио | TensorFlow | ~1 ГБ | 2–3с |
| Изображение | PyTorch | ~1 ГБ | <2с |
| Видео | YOLOv8n | ~2 ГБ | в реальном времени |
| LLM | llama.cpp (3B Q4_K_M) | ~5–6 ГБ | интерактивно |

---

### 👤 Автор
**Ислам Мубараков**  
Студент 3 курса, Программная инженерия, КФУ  
Курс: *SE 25/26 — Task Series 1*
