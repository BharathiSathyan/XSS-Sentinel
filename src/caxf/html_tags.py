from bs4 import BeautifulSoup
from sklearn.feature_extraction import DictVectorizer

class HTMLTagExtractor:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)

    def extract(self, payloads):
        tag_dicts = []

        for payload in payloads:
            soup = BeautifulSoup(payload, "lxml")
            tags = {}
            for tag in soup.find_all():
                tag_name = tag.name
                tags[tag_name] = tags.get(tag_name, 0) + 1
            tag_dicts.append(tags)

        return self.vectorizer.fit_transform(tag_dicts)
