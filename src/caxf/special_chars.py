from sklearn.feature_extraction import DictVectorizer

SPECIAL_CHARS = ['<', '>', '"', "'", '(', ')', ';', '=', '/', '\\']

class SpecialCharExtractor:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)

    def extract(self, payloads):
        char_dicts = []

        for payload in payloads:
            counts = {}
            for ch in SPECIAL_CHARS:
                count = payload.count(ch)
                if count > 0:
                    counts[ch] = count
            char_dicts.append(counts)

        return self.vectorizer.fit_transform(char_dicts)
