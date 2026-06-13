import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import re
import spacy
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="centered",
)

# ─────────────────────────────────────────
# Custom CSS — dark card aesthetic, mono accent
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

h1 {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #6b7280;
    margin-bottom: 2rem;
    letter-spacing: 0.04em;
}

.stTextArea textarea {
    background-color: #161b27 !important;
    border: 1.5px solid #2a3244 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    padding: 14px !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.25) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Result card */
.result-card {
    background: #161b27;
    border: 1px solid #2a3244;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-top: 1.4rem;
}
.emotion-label {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.confidence-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 1.4rem;
}

/* Bar rows */
.bar-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.55rem;
    gap: 0.7rem;
}
.bar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #cbd5e1;
    width: 80px;
    flex-shrink: 0;
}
.bar-track {
    flex: 1;
    background: #1e2535;
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}
.bar-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #6b7280;
    width: 42px;
    text-align: right;
    flex-shrink: 0;
}

.divider {
    border: none;
    border-top: 1px solid #2a3244;
    margin: 1.2rem 0;
}

/* Upload hint */
.hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #4b5563;
    margin-top: 0.5rem;
}

/* Badges */
.badge {
    display: inline-block;
    background: #1e2535;
    border: 1px solid #2a3244;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #7c3aed;
    margin-right: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Emotion metadata
# ─────────────────────────────────────────
EMOTIONS = {
    0: ("sadness",  "😢", "#6366f1"),
    1: ("joy",      "😄", "#10b981"),
    2: ("love",     "❤️",  "#ec4899"),
    3: ("anger",    "😠", "#ef4444"),
    4: ("fear",     "😨", "#f59e0b"),
    5: ("surprise", "😲", "#06b6d4"),
}


# ─────────────────────────────────────────
# Model definition  (must match training code)
# ─────────────────────────────────────────
class NET(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix,
                 hidden_dim=64, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=0.3, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        combined = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(combined)


# ─────────────────────────────────────────
# Load model checkpoint (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_model(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    net = NET(
        vocab_size       = ckpt["vocab_size"],
        embedding_dim    = ckpt["embedding_dim"],
        embedding_matrix = ckpt["embedding_matrix"],
        hidden_dim       = ckpt["hidden_dim"],
        n_layers         = ckpt["n_layers"],
    ).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    return net, ckpt["word2idx"], device


# ─────────────────────────────────────────
# spaCy (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])


# ─────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────
def preprocess(text: str, nlp, word2idx: dict, max_len: int = 200):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    tokens = [
        t.lemma_ for t in doc
        if not t.is_stop and not t.is_punct
        and not t.is_space and len(t.lemma_) > 1
    ]

    seq = [word2idx.get(t, 1) for t in tokens]
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]

    return torch.tensor([seq], dtype=torch.long)


# ─────────────────────────────────────────
# Predict
# ─────────────────────────────────────────
def predict(text, model, nlp, word2idx, device):
    inp = preprocess(text, nlp, word2idx).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top = int(np.argmax(probs))
    return top, probs


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.markdown("<h1>Emotion Detector</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">BiLSTM · Word2Vec · dair-ai/emotion</p>',
    unsafe_allow_html=True
)

# ── Model upload ──
uploaded = st.file_uploader(
    "Upload your `emotion_model.pth` checkpoint",
    type=["pth"],
    label_visibility="collapsed",
)
st.markdown('<p class="hint">↑ drag & drop your emotion_model.pth here</p>', unsafe_allow_html=True)

if uploaded:
    # Save to a temp path so torch.load can open it
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
        f.write(uploaded.read())
        tmp_path = f.name

    try:
        model, word2idx, device = load_model(tmp_path)
        nlp = load_spacy()

        st.markdown(
            f'<span class="badge">device: {"cuda" if device.type=="cuda" else "cpu"}</span>'
            f'<span class="badge">vocab: {len(word2idx):,}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Text input ──
        text_input = st.text_area(
            "Enter text",
            placeholder="Type something and see how it feels…",
            height=130,
            label_visibility="collapsed",
        )

        if st.button("Detect emotion"):
            if not text_input.strip():
                st.warning("Please type something first.")
            else:
                with st.spinner("Thinking…"):
                    top_idx, probs = predict(text_input, model, nlp, word2idx, device)

                name, icon, color = EMOTIONS[top_idx]
                confidence = probs[top_idx]

                st.markdown(f"""
                <div class="result-card">
                    <div class="emotion-label" style="color:{color};">{icon} {name.capitalize()}</div>
                    <div class="confidence-text">confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                for idx in range(6):
                    ename, eicon, _ = EMOTIONS[idx]
                    pct = float(probs[idx])
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{eicon} {ename}**")
                        st.progress(pct)
                    with col2:
                        st.markdown(f"<div style='padding-top:1.6rem; font-family:monospace; color:#9ca3af;'>{pct:.1%}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to load model: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
else:
    # Placeholder state
    st.markdown("""
    <div class="result-card" style="text-align:center; padding: 2.5rem;">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">🧠</div>
        <div style="color:#4b5563; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
            Upload your checkpoint to get started
        </div>
    </div>
    """, unsafe_allow_html=True)