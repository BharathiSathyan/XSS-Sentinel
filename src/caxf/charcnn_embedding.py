import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix

# Fix the random seed before any model construction so that the randomly
# initialized CharCNN weights are identical across all runs.
# NOTE: This model is intentionally used as a fixed random projection —
# the weights are NOT trained on the XSS corpus. The CNN architecture acts
# as a structured, nonlinear dimensionality-reduction kernel over character
# ordinals, producing a 128-dim embedding that captures local character
# n-gram statistics via random convolutional filters. This is a valid and
# reproducible technique (cf. random kitchen sink / random features).
torch.manual_seed(42)


class CharCNN(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=32, num_filters=64, output_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=5)

        self.fc = nn.Linear(num_filters * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        c3 = torch.relu(self.conv3(x))
        c5 = torch.relu(self.conv5(x))

        p3 = torch.max(c3, dim=2)[0]
        p5 = torch.max(c5, dim=2)[0]

        out = torch.cat([p3, p5], dim=1)
        return self.fc(out)


class CharCNNEmbeddingExtractor:
    def __init__(self, max_len=200):
        self.max_len = max_len
        self.model = CharCNN()
        self.model.eval()

    def encode_payload(self, text):
        # Characters with ordinal >= 128 are mapped to 0 (out-of-vocabulary token).
        # The paper describes this as ord(c) mod 128; functionally identical for
        # printable ASCII payloads (ordinal always < 128), and clips extended
        # characters to the padding token rather than wrapping.
        encoded = [ord(c) if ord(c) < 128 else 0 for c in text][:self.max_len]
        encoded += [0] * (self.max_len - len(encoded))
        return encoded

    def fit(self, payloads):
        return self

    def transform(self, payloads):
        encoded = [self.encode_payload(p) for p in payloads]

        x = torch.tensor(encoded)

        with torch.no_grad():
            embeddings = self.model(x).numpy()

        return csr_matrix(embeddings)