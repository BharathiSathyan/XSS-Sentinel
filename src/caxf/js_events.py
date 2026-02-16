import re
from sklearn.feature_extraction import DictVectorizer

JS_EVENT_PATTERN = re.compile(r'on\w+\s*=', re.IGNORECASE)

class JSEventExtractor:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)

    def extract(self, payloads):
        event_dicts = []

        for payload in payloads:
            events = {}
            matches = JS_EVENT_PATTERN.findall(payload)
            for match in matches:
                events[match.lower()] = events.get(match.lower(), 0) + 1
            event_dicts.append(events)

        return self.vectorizer.fit_transform(event_dicts)
