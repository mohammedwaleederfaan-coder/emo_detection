# 🧠 Emotion Detection with BiLSTM + Word2Vec

A deep learning pipeline that classifies text into 6 emotions using a Bidirectional LSTM trained on pretrained Word2Vec embeddings — with a Streamlit app for interactive inference.

---

## 📌 Overview

This project fine-tunes a BiLSTM model on the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset to detect the following emotions from short English text:

| Label | Emotion     |
| ----- | ----------- |
| 0     | 😢 Sadness  |
| 1     | 😄 Joy      |
| 2     | ❤️ Love     |
| 3     | 😠 Anger    |
| 4     | 😨 Fear     |
| 5     | 😲 Surprise |

---

## 🗂️ Project Structure

```
emotion-detection/
│
├── emotion_classifier.py   # Full training pipeline
├── emotion_app.py          # Streamlit inference app
├── emotion_model.pth       # Saved model checkpoint (generated after training)
└── README.md
```

---

## ⚙️ Pipeline

```
Raw Text
   ↓
Text Cleaning (regex)
   ↓
spaCy Lemmatization + Stopword Removal
   ↓
Word2Vec Training (Skip-Gram, 100d)
   ↓
Sequence Padding (max_len=200)
   ↓
BiLSTM (2 layers, hidden=64, bidirectional)
   ↓
Classifier Head (Linear → ReLU → Dropout × 2)
   ↓
CrossEntropyLoss + AdamW
```

---

## 🏗️ Model Architecture

```
Embedding(vocab_size, 100)   ← initialized from Word2Vec
      ↓
BiLSTM(100 → 64, layers=2, dropout=0.3)
      ↓
Concat [hidden_fwd, hidden_bwd]  →  128-dim vector
      ↓
Linear(128 → 128) → ReLU → Dropout(0.4)
Linear(128 → 64)  → ReLU → Dropout(0.3)
Linear(64  → 6)
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Train the model

```bash
python emotion_classifier.py
```

This will train for 10 epochs and save `emotion_model.pth`.

### 3. Run the Streamlit app

```bash
streamlit run emotion_app.py
```

Upload `emotion_model.pth` in the app, type any text, and click **Detect emotion**.

---

## 📊 Training Details

| Hyperparameter | Value     |
| -------------- | --------- |
| Embedding dim  | 100       |
| Hidden dim     | 64        |
| LSTM layers    | 2         |
| Bidirectional  | ✅         |
| Dropout        | 0.3 / 0.4 |
| Optimizer      | AdamW     |
| Learning rate  | 0.001     |
| Weight decay   | 1e-4      |
| Batch size     | 32        |
| Epochs         | 10        |
| Max seq length | 200       |
| Grad clipping  | 1.0       |

---

## 🔧 Known Limitations

- The model tends to **over-predict Sadness and Fear** due to class imbalance in the dataset.
- Word2Vec produces **static embeddings** — it doesn't capture context (e.g., "not happy" vs "happy" map to similar vectors).





---

## 👤 Author

**Mohammed Waleed**

- GitHub: [@mohammedwaleederfaan-coder](https://github.com/mohammedwaleederfaan-coder)
- LinkedIn: [mohammed-waleed](https://linkedin.com/in/mohammed-waleed-0065b9409)