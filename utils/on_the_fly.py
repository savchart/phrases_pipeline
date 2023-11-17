from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from utils.phrases_processor import phrases_vectors, get_phrases, phrase_vector

model_path = Path(__file__).parent.parent / 'data' / 'w2v.txt'


class OnTheFlyProcessor:
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format(str(model_path), binary=True)
        self.phrases_vectors = phrases_vectors(get_phrases(), self.model)

    def on_the_fly_execution(self, string):
        string_vector = phrase_vector(string, self.model)
        max_sim = 0
        max_phrase = ''
        for phrase in self.phrases_vectors:
            sim = cosine_similarity([string_vector], [self.phrases_vectors[phrase]])[0][0]
            if sim > max_sim:
                max_sim = sim
                max_phrase = phrase
        return max_phrase, max_sim



