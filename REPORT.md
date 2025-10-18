# 📑 REPORT — Machine Learning Task Series 1  
**Author:** Islam Mubarakov  
**Course:** SE 25/26  
**University:** Kazan Federal University  
**Platform:** macOS (Apple Silicon, M2, 8GB RAM)

---

## 🇬🇧 English Version

### 🎯 Project Objective
The main goal of this work is to explore the practical use of ready-made AI/ML/DL models to solve applied problems in various domains — text, audio, image, and video — as well as to deploy a local large language model (LLM) on a personal computer without the use of cloud services.

This demonstrates how modern AI frameworks (Hugging Face, TensorFlow, PyTorch, Ultralytics, and llama.cpp) allow us to apply advanced deep learning models with minimal code and no retraining.

---

### 🧩 Implemented Tasks

| № | Task Type | Library | Model | Dataset / Source | Description |
|---|------------|----------|--------|------------------|-------------|
| 1 | Text Sentiment Analysis | Hugging Face Transformers | distilbert-base-uncased-finetuned-sst-2-english | SST-2 (Stanford Sentiment Treebank) | Determines positive/negative text tone |
| 2 | Audio Classification | TensorFlow Hub | YAMNet | AudioSet (Google) | Detects sound types (speech, animals, noise) |
| 3 | Image Classification | PyTorch | ResNet50 | ImageNet | Classifies objects in an image |
| 4 | Object Detection in Video | Ultralytics YOLOv8 | yolov8n.pt | COCO dataset | Detects and labels objects in video frames |
| 5 | Local LLM | llama.cpp / llama-cpp-python | Llama 3.2 3B Instruct (GGUF) | — | Generates and processes text offline |

---

### ⚙️ Frameworks and Their Advantages

| Framework | Advantages |
|------------|-------------|
| Hugging Face Transformers | Simple API for NLP tasks, thousands of pre-trained models, excellent documentation |
| TensorFlow Hub | Modular system for loading pre-trained neural networks, optimized for performance |
| PyTorch | Flexible and powerful deep learning framework with GPU/MPS support |
| Ultralytics YOLOv8 | Fast object detection, real-time processing, easy integration |
| llama.cpp | Lightweight and efficient framework for running LLMs locally without cloud dependencies |

---

### 📊 Model Evaluation Metrics

| Metric | Meaning | Usage Example |
|--------|----------|----------------|
| Accuracy | Percentage of correct predictions | Text sentiment, image classification |
| Precision | True positives / (True + False positives) | Object detection, audio classification |
| Recall | True positives / (True + False negatives) | YOLO detection performance |
| F1-score | Harmonic mean of Precision and Recall | Balanced evaluation for detection tasks |
| Latency | Time required for inference (seconds) | Real-time performance estimation |
| GPU / CPU Usage | Hardware utilization | Monitored during model execution (MPS on macOS) |

---

### ⚡ Efficiency Analysis (Measured on MacBook Air M2)

| Task | Runtime | Memory Usage | Observations |
|------|----------|--------------|---------------|
| Text Analysis | <1 sec | ~0.5 GB | Instant classification |
| Audio Classification | 2–3 sec | ~1 GB | Moderate load |
| Image Classification | 1–2 sec | ~1 GB | Smooth inference |
| Video Detection | Real-time | ~2 GB | Efficient processing |
| LLM Chat | 2–3 sec per token | ~6 GB | Stable interactive chat |

---

### 🧠 Principles of Operation

#### 1️⃣ Text Sentiment (Hugging Face)
- Tokenizes input text into subwords  
- Passes through DistilBERT transformer layers  
- Outputs a probability distribution (positive / negative)

#### 2️⃣ Audio (YAMNet)
- Converts audio waveform → log-mel spectrogram  
- Processes through pretrained MobileNet-based network  
- Returns top sound class probabilities

#### 3️⃣ Image (ResNet50)
- Divides image into feature maps  
- Extracts hierarchical representations via convolutional blocks  
- Classifies using a fully connected layer over ImageNet labels

#### 4️⃣ Video (YOLOv8)
- Splits video into frames  
- Applies object detection per frame  
- Draws bounding boxes with class and confidence score

#### 5️⃣ LLM (Llama.cpp)
- Uses transformer architecture with self-attention  
- Predicts next token based on context  
- Entire model runs locally in quantized format (GGUF)

---

### 🧮 Dataset Structures

| Task | Dataset | Type | Size / Example |
|------|----------|------|----------------|
| Text | SST-2 | Text corpus | Short English sentences |
| Audio | AudioSet | Sound clips | WAV, 16kHz mono |
| Image | ImageNet | Images | JPEG 224×224 |
| Video | COCO (via YOLO) | Video frames | MP4 |
| LLM | — | Text input | Prompt / Response pairs |

---

### 🧠 Model Precision and Quality Measurement

- For classification models (text, image, audio), accuracy and F1-score were measured on example samples.  
- For YOLO, detection precision ≈ 0.8–0.9 based on confidence score threshold 0.4.  
- For LLM, qualitative evaluation (relevance, fluency, latency) was applied.

---

### 🧩 Implementation Features

- All models work fully locally, without cloud access.  
- Code optimized for Apple MPS (Metal Performance Shaders).  
- No fine-tuning required — all models use pretrained weights.  
- Each task runs independently and can be executed via Python CLI.

---

### ✅ Conclusions

This project demonstrates that:
1. Modern AI frameworks make it possible to use deep learning in real-world tasks without retraining.  
2. macOS (Apple Silicon) efficiently handles complex models locally.  
3. LLMs like Llama 3.2 can operate offline and serve as a foundation for local intelligent assistants.

---

## 🇷🇺 Русская версия

### 🎯 Цель проекта
Главная цель — продемонстрировать практическое применение готовых моделей машинного обучения (AI/ML/DL) для решения прикладных задач по анализу текста, звука, изображений и видео, а также развёртывание локальной языковой модели (LLM) без использования облачных сервисов.

---

### 🧩 Реализованные задачи

| № | Тип задачи | Библиотека | Модель | Датасет / источник | Описание |
|---|-------------|-------------|---------|--------------------|-----------|
| 1 | Анализ тональности текста | Hugging Face Transformers | distilbert-base-uncased-finetuned-sst-2-english | SST-2 | Определяет позитивную или негативную окраску текста |
| 2 | Классификация звуков | TensorFlow Hub | YAMNet | AudioSet | Распознаёт звуки (голос, животные, предметы) |
| 3 | Классификация изображений | PyTorch | ResNet50 | ImageNet | Определяет объект на изображении |
| 4 | Детекция объектов в видео | Ultralytics YOLOv8 | yolov8n.pt | COCO | Находит и помечает объекты в кадрах видео |
| 5 | Локальная LLM | llama.cpp / llama-cpp-python | Llama 3.2 3B Instruct (GGUF) | — | Работает локально без интернета |

---

### ⚙️ Используемые фреймворки и их преимущества

| Библиотека | Преимущества |
|-------------|--------------|
| Hugging Face | Простота использования, готовые модели, отличная документация |
| TensorFlow Hub | Модульная система загрузки нейросетей |
| PyTorch | Гибкость, поддержка MPS и GPU |
| Ultralytics YOLOv8 | Скорость, простота, real-time |
| llama.cpp | Локальное исполнение без облака, малая нагрузка |

---

### 📊 Метрики качества

| Метрика | Описание | Применение |
|----------|-----------|-------------|
| Accuracy | Доля верных предсказаний | Классификация |
| Precision | TP / (TP + FP) | Детекция объектов |
| Recall | TP / (TP + FN) | YOLO |
| F1-score | Баланс точности и полноты | Общая оценка |
| Latency | Время отклика модели | В реальном времени |
| GPU Usage | Нагрузка на устройство | Оптимизация MPS |

---

### ⚡ Эффективность (MacBook Air M2)

| Задача | Время | Память | Комментарий |
|--------|--------|--------|--------------|
| Текст | <1 сек | ~0.5 ГБ | Очень быстро |
| Аудио | 2–3 сек | ~1 ГБ | Средняя нагрузка |
| Изображения | 1–2 сек | ~1 ГБ | Стабильно |
| Видео | Real-time | ~2 ГБ | Быстрая обработка |
| LLM | 2–3 сек на токен | ~6 ГБ | Устойчивая работа |

---

### 🧠 Принцип работы моделей

#### 1️⃣ Анализ тональности текста (Hugging Face / DistilBERT)
- Исходный текст разбивается на токены (подслова).  
- Токены подаются в трансформерную модель DistilBERT, обученную на корпусе SST-2.  
- Модель вычисляет эмбеддинги и определяет, относится ли фраза к позитивной или негативной категории.  
- Результат выводится как метка (POSITIVE / NEGATIVE) и вероятность (score).  

#### 2️⃣ Классификация звука (TensorFlow Hub / YAMNet)
- Аудиосигнал читается в виде временного ряда (waveform).  
- Он преобразуется в лог-мел-спектрограмму, отражающую частотное распределение звука.  
- Спектрограмма передаётся в сверточную сеть MobileNet, обученную на датасете AudioSet.  
- На выходе модель выдаёт вероятности по 521 категории (речь, лай, аплодисменты и т.д.).  
- Выводится топ-5 наиболее вероятных классов.  

#### 3️⃣ Классификация изображений (PyTorch / ResNet50)
- Изображение нормализуется и приводится к размеру 224×224.  
- Проходит через свёрточные блоки ResNet50, которые извлекают многослойные признаки.  
- Модель использует остаточные соединения (skip-connections), что ускоряет и стабилизирует обучение.  
- Финальный слой вычисляет распределение вероятностей по 1000 категориям (ImageNet).  
- На экран выводится наиболее вероятный класс объекта.  

#### 4️⃣ Детекция объектов в видео (Ultralytics YOLOv8)
- Видео разбивается на отдельные кадры.  
- Для каждого кадра YOLOv8 выполняет обнаружение объектов:  
  - прогнозирует bounding boxes,  
  - определяет классы (cat, person, car и др.),  
  - вычисляет confidence score (уверенность).  
- На кадре отображаются контуры объектов и подписи.  
- Модель работает в режиме реального времени на Apple MPS.  

#### 5️⃣ Локальная языковая модель (Llama.cpp / Llama 3.2 3B)
- Используется архитектура Transformer с механизмом самовнимания (Self-Attention).  
- Модель принимает текстовый запрос (prompt) и предсказывает следующий токен, опираясь на контекст.  
- Генерация продолжается пошагово, формируя осмысленный ответ.  
- Модель работает локально на MacBook Air M2 в формате GGUF, что позволяет запускать её без интернета.  
- Используется квантованная версия (Q4_K_M), оптимизированная под устройства с 8 ГБ ОЗУ.  

---

### ✅ Выводы

В ходе работы показано, что:
1. Современные AI-библиотеки позволяют использовать сложные модели без обучения.  
2. Apple Silicon обеспечивает стабильную производительность при локальном запуске.  
3. LLM (Llama 3.2) успешно работает без подключения к интернету.

---

### 📅 Date: October 2025  
**Project completed by:** Islam Mubarakov  
Kazan Federal University, Institute of ITIS  
