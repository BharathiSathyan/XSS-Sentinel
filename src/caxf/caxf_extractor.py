from scipy.sparse import hstack

from .html_tags import HTMLTagExtractor
from .js_events import JSEventExtractor
from .url_features import URLFeatureExtractor
from .special_chars import SpecialCharExtractor
from .keywords import KeywordExtractor
from .tfidf_char import CharTFIDFExtractor


class CAXFExtractor:
    def __init__(self):
        self.html_extractor = HTMLTagExtractor()
        self.js_extractor = JSEventExtractor()
        self.url_extractor = URLFeatureExtractor()
        self.sc_extractor = SpecialCharExtractor()
        self.kw_extractor = KeywordExtractor()
        self.tfidf_extractor = CharTFIDFExtractor()

    def fit_transform(self, payloads):
        print("Extracting HTML tag features...")
        f_tags = self.html_extractor.extract(payloads)

        print("Extracting JS event features...")
        f_events = self.js_extractor.extract(payloads)

        print("Extracting URL features...")
        f_url = self.url_extractor.extract(payloads)

        print("Extracting special character features...")
        f_sc = self.sc_extractor.extract(payloads)

        print("Extracting keyword features...")
        f_kw = self.kw_extractor.extract(payloads)

        print("Extracting TF-IDF features...")
        f_tfidf = self.tfidf_extractor.extract(payloads)

        print("Combining all features...")
        embedding = hstack([f_tags, f_events, f_url, f_sc, f_kw, f_tfidf])

        return embedding
