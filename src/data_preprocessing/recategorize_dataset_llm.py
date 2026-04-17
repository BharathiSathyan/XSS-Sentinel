import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()


# Prompt 

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


def classify_payload(payload):
    inputs = tokenizer(
        build_prompt(payload),
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# Re-categorization with checkpoints
def recategorize_csv(
    input_csv="data/raw/XSS_dataset.csv",
    output_csv="data/processed/xss_llm_multiclass.csv",
    checkpoint_interval=100
):
    # Load input
    df = pd.read_csv(input_csv)

    # Resume if output exists
    if os.path.exists(output_csv):
        print("🔄 Resuming from checkpoint...")
        df_out = pd.read_csv(output_csv)

        if "final_label" not in df_out.columns:
            raise ValueError("Checkpoint file missing 'final_label' column")

        df["final_label"] = df_out["final_label"]
    else:
        df["final_label"] = None

    for idx, row in df.iterrows():
        # Skip already processed rows
        if pd.notna(df.at[idx, "final_label"]):
            continue

        payload = row["Sentence"]
        original_label = row["Label"]

        if original_label == 0:
            df.at[idx, "final_label"] = "Normal"
        else:
            df.at[idx, "final_label"] = classify_payload(payload)

        # Logging
        if idx % checkpoint_interval == 0:
            print(f"Row {idx} → {df.at[idx, 'final_label']}")

            # Save checkpoint
            df[["Sentence", "final_label"]].to_csv(output_csv, index=False)
            print(" Checkpoint saved")

    # Final save
    df[["Sentence", "final_label"]].to_csv(output_csv, index=False)

    print(" LLM re-categorization complete")
    print(df["final_label"].value_counts())



if __name__ == "__main__":
    recategorize_csv()