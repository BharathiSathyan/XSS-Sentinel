import numpy as np
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer


class SentenceEmbeddingExtractor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("Loading Sentence Transformer model...")
        self.model = SentenceTransformer(model_name)

    def fit(self, payloads):
        # Sentence transformers do not require training
        return self

    def transform(self, payloads):
        embeddings = self.model.encode(
            payloads.tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64
        )

        return csr_matrix(embeddings)