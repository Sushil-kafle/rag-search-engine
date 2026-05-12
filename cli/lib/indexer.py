import string
from nltk.stem import PorterStemmer
import pickle
import os
from pathlib import Path
from collections import Counter, defaultdict

from utils.load_data import load_stop_words


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STOP_WORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"


class InvertedIndex:
    def __init__(self):
        self.index: dict = {}
        self.docmap: dict = {}
        self.term_frequencies: dict = defaultdict(Counter)

    def __add_document(self, doc_id, text):

        # remove punctiation
        punc = string.punctuation
        text = text.translate(str.maketrans("", "", punc))

        # make lowercase

        text = text.lower()

        # tokenize
        tokenized_text = text.split()

        # remove stop words and stem
        stemmer = PorterStemmer()
        stop_words = load_stop_words(STOP_WORDS_PATH)
        filtered_tokens = [
            stemmer.stem(token) for token in tokenized_text if token not in stop_words
        ]

        # make index
        for token in filtered_tokens:
            if token not in self.index:
                self.index[token] = []

            self.index[token].append(doc_id)

            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        term = term.lower()

        return sorted(self.index.get(term, []))

    def build(self, movies):
        for movie in movies.get("movies"):
            title = movie.get("title")
            id = movie.get("id")
            description = movie.get("description")

            combined_text = f"{title} {description}"

            self.__add_document(id, combined_text)

            self.docmap[id] = movie

    def save(self, path):
        os.makedirs(name=path, exist_ok=True)

        with open(f"{path}/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open(f"{path}/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open(f"{path}/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self, path):
        try:
            with open(f"{path}/index.pkl", "rb") as f:
                self.index = pickle.load(f)
        except FileNotFoundError as e:
            print(f"index not found {e}")
            self.index = {}

        try:
            with open(f"{path}/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
        except FileNotFoundError as e:
            print(f"docmap not found {e}")
            self.docmap = {}

        try:
            with open(f"{path}/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError as e:
            print(f"term_frequencies not found {e}")
            self.term_frequencies = defaultdict(Counter)
