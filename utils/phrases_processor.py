import re
import numpy as np
from pathlib import Path

phrases_path = Path(__file__).parent.parent / 'data' / 'phrases.csv'


def get_word_set():
    word_list = []
    with open(phrases_path, 'r', encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.lower()
            line = re.sub(r'[^a-z ]', '', line)
            word_list += [word for word in line.split()]
    return set(word_list[1:])


def get_phrases():
    phrases = []
    with open(phrases_path, 'r', encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.lower()
            line = re.sub(r'[^a-z ]', '', line)
            phrases.append(line)
    return set(phrases[1:])


def phrase_vector(phrase, model):
    sum_vector = 0
    for word in phrase.split():
        if word not in model:
            continue
        sum_vector += model[word]
    norm = np.linalg.norm(sum_vector)
    if norm == 0:
        return sum_vector
    return sum_vector / norm


def phrases_vectors(phrases, model):
    return {phrase: phrase_vector(phrase, model) for phrase in phrases}
