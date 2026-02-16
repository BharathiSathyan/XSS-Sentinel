import pandas as pd
df = pd.read_csv("data/processed/xss_llm_multiclass.csv")

print(df["final_label"].value_counts())
print(df["final_label"].unique())