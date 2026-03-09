import pandas as pd
from sklearn.model_selection import train_test_split
import time

# tfidf
from src.caxf.caxf_extractor import CAXFExtractor
# sentence embedding
# from src.caxf.caxf_extractor_sentence_embedding import CAXFExtractor

# Load dataset
df = pd.read_csv("data/processed/Final_XSS_4class_BALANCED_dataset.csv")

print("Dataset loaded.")
print("Total samples:", len(df))


X = df["Sentence"].astype(str)
y = df["Final_Label"]  

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# Initialize CAXF
caxf = CAXFExtractor()

start = time.time()
# Fit ONLY on training data
print("\nFitting CAXF on training data...")
caxf.fit(X_train)

#  Transform train & test
print("\nTransforming training data...")
X_train_embed = caxf.transform(X_train)

print("\nTransforming test data...")
X_test_embed = caxf.transform(X_test)
end = time.time()
#  Basic checks
print("\nEmbedding shapes:")
print("Train:", X_train_embed.shape)
print("Test :", X_test_embed.shape)

print("\nNon-zero counts (train first 5 rows):")
print(X_train_embed[:5].getnnz(axis=1))
print("Time taken:", round(end - start, 2), "seconds")

print("Time taken:", round(end - start, 2), "seconds")