from process import Preprocess

import itertools
import random
import numpy as np

class ConEx_token(object):
    """docstring for ConEx_token"""
    def __init__(self, algorithms):
        super(ConEx_token, self).__init__()
        self.html = None
        self.tokens = None
        self.algorithms = algorithms
        self.labels = None
        self.probs = None
        self.combined_label = None
        self.combined_prob = None

    def process_html(self, html, tidy=True, body_only=False):
        self.tokens = Preprocess.get_all_tokens(html, tidy, body_only)
        for algorithm in self.algorithms:
            algorithm.process_html_part(html, self.tokens)

    def check(self):
        if self.html == None:
            raise Exception("Please feed a html string first (by calling \"process_html\" method).")

    def predict_token_label(self):
        self.labels = [algorithm.predict_token_label() for algorithm in self.algorithms]
        min_len = min([len(single_ret) for single_ret in self.labels])
        self.labels = [single_ret[:min_len] for single_ret in self.labels]
        self.combined_label = (sum(self.labels) + np.random.choice([1e-7, -1e-7], min_len))/len(self.labels) > 0.5
        return self.combined_label

    def predict_token_prob(self, merge='avg'):
        self.probs = [algorithm.predict_token_prob() for algorithm in self.algorithms]
        min_len = min([len(single_prob) for single_prob in self.probs])
        self.probs = [single_prob[:min_len] for single_prob in self.probs]
        if merge == 'avg':
            self.combined_prob = np.average(self.probs, axis=0)
        elif merge == 'max':
            self.combined_prob = np.max(self.probs, axis=0)
        else:
            raise Exception("Wrong merge argument: " + merge)
        return self.combined_prob

    def filter_content(self, merge='vote'):
        words = filter(lambda token:not(token.startswith('<') and token.endswith('>')),self.tokens)
        if merge == 'vote':
            ret = list(itertools.compress(words, self.combined_label))
        else:
            self.predict_token_prob(merge)
            ret = list(itertools.compress(words, self.combined_prob > 0.5))
        return ret

    def filter_content_all(self):
        words = filter(lambda token:not(token.startswith('<') and token.endswith('>')),self.tokens)
        ret = {self.algorithms[i].__class__.__name__:list(itertools.compress(words, pred)) 
                        for i, pred in enumerate(self.labels)}
        if len(self.algorithms) > 1:
            ret['token_combined' + '-vote'] = self.filter_content('vote')
            ret['token_combined' + '-avg'] = self.filter_content('avg')
            ret['token_combined' + '-max'] = self.filter_content('max')
        return ret