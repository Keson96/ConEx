from CTTD import CTTD
from CETR import CETR
from process import Preprocess

import itertools
import numpy as np

class ConEx_line(object):
    """docstring for ConEx_line"""
    def __init__(self, algorithms):
        super(ConEx_line, self).__init__()
        self.html = None
        self.tokens_by_line = None
        self.algorithms = algorithms
        self.labels = None
        self.combined_label = None
        self.probs = None
        self.combined_prob = None

    def process_html(self, html, tidy=True, body_only=False):
        self.html = html
        self.tokens_by_line = Preprocess.get_tokens_by_line(html, tidy, body_only)
        for algorithm in self.algorithms:
            algorithm.process_html_part(html, self.tokens_by_line)

    def check(self):
        if self.html == None:
            raise Exception("Please feed a html string first (by calling \"process_html\" method).")

    def predict_line_label(self):
        self.labels = [algorithm.predict_line_label() for algorithm in self.algorithms]
        min_len = min([len(single_ret) for single_ret in self.labels])
        self.labels = [single_ret[:min_len].astype('int') for single_ret in self.labels]
        self.combined_label = (sum(self.labels) + np.random.choice([1e-7, -1e-7], min_len))/len(self.labels) > 0.5
        return self.combined_label

    def predict_line_prob(self, merge='avg'):
        self.probs = [algorithm.predict_line_prob() for algorithm in self.algorithms]
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
        if merge == 'vote':
            ret = [token for line in itertools.compress(self.tokens_by_line, self.combined_label) for token in line
                    if (not (token.startswith('<') and token.endswith('>')) and token.strip())]
        else:
            self.predict_line_prob(merge)
            ret = [token for line in itertools.compress(self.tokens_by_line, self.combined_prob > 0.5) for token in line
                    if (not (token.startswith('<') and token.endswith('>')) and token.strip())]
        return ret

    def filter_content_all(self):
        ret = {self.algorithms[i].__class__.__name__ + self.algorithms[i].method 
                        if hasattr(self.algorithms[i], 'method') else self.algorithms[i].__class__.__name__:
                    [token for line in itertools.compress(self.tokens_by_line, pred) for token in line
                        if (not (token.startswith('<') and token.endswith('>')) and token.strip())]
                    for i, pred in enumerate(self.labels)}
        if len(self.algorithms) > 1:
            ret['line_combined' + '-vote'] = self.filter_content('vote')
            ret['line_combined' + '-avg'] = self.filter_content('avg')
            ret['line_combined' + '-max'] = self.filter_content('max')

        return ret


        