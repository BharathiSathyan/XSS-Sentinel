from sklearn.feature_extraction import DictVectorizer

XSS_KEYWORDS = [
    "script", "alert", "eval", "document", "window",
    "cookie", "location", "onerror", "onload",
    "iframe", "img", "svg"
]

class KeywordExtractor:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)

    def extract(self, payloads):
        keyword_dicts = []

        for payload in payloads:
            payload_lower = payload.lower()
            counts = {}

            for kw in XSS_KEYWORDS:
                count = payload_lower.count(kw)
                if count > 0:
                    counts[kw] = count

            keyword_dicts.append(counts)

        return self.vectorizer.fit_transform(keyword_dicts)
