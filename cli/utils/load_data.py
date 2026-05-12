import json


def load_movies(path):
    with open(path, "r") as f:
        return json.load(f)


def load_stop_words(path):
    with open(path, "r") as f:
        return f.read().splitlines()
