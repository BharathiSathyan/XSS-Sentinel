import warnings
from bs4 import MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning
from bs4 import BeautifulSoup
from sklearn.feature_extraction import DictVectorizer
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

class HTMLTagExtractor:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)

    def extract(self, payloads):
        tag_dicts = []

        for payload in payloads:
            try:
                soup = BeautifulSoup(payload, "lxml")
            except:
                soup = BeautifulSoup(payload, "html.parser")
            tags = {}
            for tag in soup.find_all():
                tag_name = tag.name
                tags[tag_name] = tags.get(tag_name, 0) + 1
            tag_dicts.append(tags)

        return self.vectorizer.fit_transform(tag_dicts)
    
    def fit(self, payloads):
        tag_dicts = []

        for payload in payloads:
            try:
                soup = BeautifulSoup(payload, "lxml")
            except:
                soup = BeautifulSoup(payload, "html.parser")
            tags = {}
            for tag in soup.find_all():
                tag_name = tag.name
                tags[tag_name] = tags.get(tag_name, 0) + 1
            tag_dicts.append(tags)

        self.vectorizer.fit(tag_dicts)
        return self

    def transform(self, payloads):
        tag_dicts = []

        for payload in payloads:
            try:
                soup = BeautifulSoup(payload, "lxml")
            except:
                soup = BeautifulSoup(payload, "html.parser")
            tags = {}
            for tag in soup.find_all():
                tag_name = tag.name
                tags[tag_name] = tags.get(tag_name, 0) + 1
            tag_dicts.append(tags)

        return self.vectorizer.transform(tag_dicts)
