import os
import pandas as pd
import re

INPUT_PATH = "data/raw/XSS_dataset.csv"
OUTPUT_PATH = "data/processed/XSS_recategorized.csv"
CHECKPOINT_INTERVAL = 100

# -----------------------------
# Pattern Definitions
# -----------------------------

DOM_PATTERNS = [
    r'document\.location',
    r'document\.referrer',
    r'window\.location',
    r'location\.hash',
    r'innerhtml\s*=',
    r'outerhtml\s*=',
]

STORED_PATTERNS = [
    r'<script.*?>',
    r'</script>',
    r'<iframe',
    r'<object',
    r'<embed',
    r'<svg',
]

REFLECTED_PATTERNS = [
    r'on\w+\s*=',
    r'%3c',
    r'%3e',
    r'alert\s*\(',
]

# -----------------------------
# Categorization Function
# -----------------------------

def categorize_xss(sentence, label):
    sentence = str(sentence).lower()

    if label == 0:
        return "Normal"

    # DOM-based
    for pattern in DOM_PATTERNS:
        if re.search(pattern, sentence):
            return "DOM-based XSS"

    # Stored
    for pattern in STORED_PATTERNS:
        if re.search(pattern, sentence):
            return "Stored XSS"

    # Reflected
    for pattern in REFLECTED_PATTERNS:
        if re.search(pattern, sentence):
            return "Reflected XSS"

    return "Stored XSS"


# -----------------------------
# Main Execution (Checkpointed)
# -----------------------------

def main():
    df = pd.read_csv(INPUT_PATH)

    # If checkpoint exists, resume
    if os.path.exists(OUTPUT_PATH):
        print("🔄 Resuming from checkpoint...")
        df_out = pd.read_csv(OUTPUT_PATH)

        if "final_label" not in df_out.columns:
            raise ValueError("Checkpoint file missing 'final_label' column")

        df["final_label"] = df_out["final_label"]
    else:
        df["final_label"] = None

    for idx, row in df.iterrows():

        # Skip already processed rows
        if pd.notna(df.at[idx, "final_label"]):
            continue

        df.at[idx, "final_label"] = categorize_xss(
            row["Sentence"],
            row["Label"]
        )

        # Checkpoint save
        if idx % CHECKPOINT_INTERVAL == 0:
            print(f"Processed row {idx}")
            df[["Sentence", "final_label"]].to_csv(
                OUTPUT_PATH,
                index=False
            )
            print("💾 Checkpoint saved")

    # Final save
    df[["Sentence", "final_label"]].to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n✅ Re-categorization complete")
    print("\nFinal class distribution:\n")
    print(df["final_label"].value_counts())


if __name__ == "__main__":
    main()