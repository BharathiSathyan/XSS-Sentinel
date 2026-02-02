import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Load FLAN-T5 (CPU-friendly)
# -----------------------------
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

# -----------------------------
# Prompt template (CRITICAL)
# -----------------------------
def build_prompt(payload):
    return f"""
Classify the following web payload into exactly one category.

Categories:
- Normal
- Stored XSS
- Reflected XSS
- DOM-based XSS

Rules:
- DOM-based XSS uses JavaScript DOM objects (document, window, innerHTML)
- Stored XSS contains HTML/script injection
- Reflected XSS uses event handlers or encoded input
- Normal is benign HTML or text

Payload:
{payload}

Answer with ONLY one category name.
"""

# -----------------------------
# LLM inference
# -----------------------------
def classify_payload(payload):
    prompt = build_prompt(payload)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10
        )

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction.strip()

# -----------------------------
# Main re-categorization
# -----------------------------
def recategorize_csv(
    input_csv="data/raw/xss_dataset.csv",
    output_csv="data/processed/xss_llm_multiclass.csv"
):
    df = pd.read_csv(input_csv)

    final_labels = []

    for _, row in df.iterrows():
        payload = row["Sentence"]
        original_label = row["Label"]

        if original_label == 0:
            final_labels.append("Normal")
        else:
            final_labels.append(classify_payload(payload))

    df["final_label"] = final_labels
    df[["Sentence", "final_label"]].to_csv(output_csv, index=False)

    print("✅ LLM re-categorization complete")
    print(df["final_label"].value_counts())

# -----------------------------
if __name__ == "__main__":
    recategorize_csv()
