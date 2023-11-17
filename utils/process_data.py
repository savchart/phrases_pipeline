import logging
from pathlib import Path
from gensim.models import KeyedVectors
from utils.phrases_processor import get_phrases, phrases_vectors
from sklearn.metrics.pairwise import cosine_similarity

vector_path = Path(__file__).parent.parent / 'data' / 'pruned.w2v.txt'
output_path = Path(__file__).parent.parent / 'output' / 'phrases_cosine.txt'


class DataProcessor:
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format(str(vector_path))
        self.assign_list = list(self.model.index_to_key)
        self.phrases_vectors = phrases_vectors(get_phrases(), self.model)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def batch_execution(self):
        self.logger.info('Batch execution started')
        cosine_sim_matrix = cosine_similarity(list(self.phrases_vectors.values()))
        with open(output_path, 'w') as file:
            for i, row in enumerate(cosine_sim_matrix):
                for j, similarity_value in enumerate(row):
                    file.write(
                        f'{list(self.phrases_vectors.keys())[i]},'
                        f'{list(self.phrases_vectors.keys())[j]},'
                        f'{similarity_value}\n'
                    )
        self.logger.info('Batch execution finished')
