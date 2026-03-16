from scipy.sparse import hstack

from .html_tags import HTMLTagExtractor
from .js_events import JSEventExtractor
from .url_features import URLFeatureExtractor
from .special_chars import SpecialCharExtractor
from .keywords import KeywordExtractor
from .charcnn_embedding import CharCNNEmbeddingExtractor


class CAXFExtractor:
    def __init__(self):
        self.html_extractor = HTMLTagExtractor()
        self.js_extractor = JSEventExtractor()
        self.url_extractor = URLFeatureExtractor()
        self.sc_extractor = SpecialCharExtractor()
        self.kw_extractor = KeywordExtractor()
        self.embedding_extractor = CharCNNEmbeddingExtractor()

    def fit(self, payloads):
        print("Fitting HTML tags...")
        self.html_extractor.fit(payloads)

        print("Fitting JS events...")
        self.js_extractor.fit(payloads)

        print("Fitting URL features...")
        self.url_extractor.fit(payloads)

        print("Fitting special characters...")
        self.sc_extractor.fit(payloads)

        print("Fitting keywords...")
        self.kw_extractor.fit(payloads)

        print("Initializing CharCNN embedding...")
        self.embedding_extractor.fit(payloads)

        return self

    def transform(self, payloads):
        print("Transforming HTML tags...")
        f_tags = self.html_extractor.transform(payloads)

        print("Transforming JS events...")
        f_events = self.js_extractor.transform(payloads)

        print("Transforming URL features...")
        f_url = self.url_extractor.transform(payloads)

        print("Transforming special characters...")
        f_sc = self.sc_extractor.transform(payloads)

        print("Transforming keywords...")
        f_kw = self.kw_extractor.transform(payloads)

        print("Generating CharCNN embeddings...")
        f_embed = self.embedding_extractor.transform(payloads)

        return hstack([f_tags, f_events, f_url, f_sc, f_kw, f_embed])