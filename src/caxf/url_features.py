import re
import numpy as np
from scipy.sparse import csr_matrix

URL_PATTERN = re.compile(r'https?://|www\.', re.IGNORECASE)

class URLFeatureExtractor:
    def extract(self, payloads):
        counts = []

        for payload in payloads:
            count = len(URL_PATTERN.findall(payload))
            counts.append([count])

        return csr_matrix(np.array(counts))
