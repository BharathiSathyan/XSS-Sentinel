from sklearn.feature_extraction.text import TfidfVectorizer

class CharTFIDFExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3,5),
            max_features=max_features
        )

    def extract(self, payloads):
        return self.vectorizer.fit_transform(payloads)
    
    def fit(self, payloads):
        self.vectorizer.fit(payloads)
        return self

    def transform(self, payloads):
        return self.vectorizer.transform(payloads)
