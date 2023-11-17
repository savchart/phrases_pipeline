import logging
from utils.phrases_processor import get_word_set
from pathlib import Path
from gensim.models import KeyedVectors

w2v_path = Path(__file__).parent.parent / 'data' / 'w2v.bin.gz'
out_model_path = Path(__file__).parent.parent / 'data' / 'w2v.txt'
out_pruned_path = Path(__file__).parent.parent / 'data' / 'pruned.w2v.txt'


class PrepareModel:
    def __init__(self):
        self.input_path = w2v_path
        self.out_pruned = out_pruned_path
        self.out_model = out_model_path
        self.limit = 1000000
        self.corpus = get_word_set()
        self.model = KeyedVectors.load_word2vec_format(str(self.input_path), binary=True, limit=self.limit)
        self.logger = logging.getLogger(__name__)

    def cut_model_path(self):
        return self.out_model.exists()

    def prune_model_path(self):
        return self.out_pruned.exists()

    def prune_model(self):
        self.logger.info('Pruning model')
        out_file = open(self.out_pruned, 'wb')
        word_presented = self.corpus.intersection(list(self.model.index_to_key))
        out_file.write('{} {}\n'.format(len(word_presented), self.model.vector_size).encode('utf-8'))
        for word in word_presented:
            out_file.write('{} {}\n'.format(word, ' '.join(map(str, self.model[word]))).encode('utf-8'))
        out_file.close()
        self.logger.info('Pruning finished')

    def cut_model(self):
        try:
            self.logger.info('Loading word2vec model from {}', self.input_path)
            self.model.save_word2vec_format(self.out_model, binary=True)
            self.logger.info('Done')
        except Exception as e:
            self.logger.error('Failed to prepare data: {}', e)
            raise e
