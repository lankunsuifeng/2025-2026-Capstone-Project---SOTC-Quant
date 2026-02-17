import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# Load FinBERT
# =========================
MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# =========================
# Example texts (you可以随便换)
# =========================
texts = [
    "Bitcoin surges sharply as ETF inflows hit record highs.",
    "Crypto market collapses after catastrophic exchange hack.",
    "Bitcoin trades slightly lower amid mixed macroeconomic signals.",
    "Investors remain uncertain as markets await Fed decision."
]

# =========================
# Inference
# =========================
with torch.no_grad():
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)

# label mapping (保险起见)
id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
label2id = {v: k for k, v in id2label.items()}

neg_i = label2id["negative"]
neu_i = label2id["neutral"]
pos_i = label2id["positive"]

# =========================
# Build sentiment signals
# =========================
rows = []
for text, p in zip(texts, probs):
    p_neg = p[neg_i].item()
    p_neu = p[neu_i].item()
    p_pos = p[pos_i].item()

    # 方向（risk-on / risk-off）
    direction = p_pos - p_neg

    # 强度 proxy（推荐这两个）
    intensity_1 = max(p_pos, p_neg)      # 最大置信度
    intensity_2 = 1.0 - p_neu             # 远离中性程度

    rows.append({
        "text": text,
        "P_neg": p_neg,
        "P_neu": p_neu,
        "P_pos": p_pos,
        "direction": direction,
        "intensity_max_prob": intensity_1,
        "intensity_1_minus_neu": intensity_2
    })

df = pd.DataFrame(rows)

print(df.round(3))
