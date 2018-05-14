from process import Preprocess

import itertools
import numpy as np

class ConEx_dom(object):
    """docstring for ConEx_dom"""
    def __init__(self, algorithms):
        super(ConEx_dom, self).__init__()
        self.html = None
        self.body = None
        self.algorithms = algorithms
        self.labels = None
        self.combined_label = None
        self.probs = None
        self.combined_prob = None

    def process_html(self, html, tidy=True, body_only=False):
        self.html = html
        root = Preprocess.get_html_nodes(self.html, tidy, body_only)
        if body_only:
            self.body = root
        else:
            self.body = root.find('body')
        for i, node in enumerate(self.body.iter()):
            node.attrib['No'] = str(i)
            for child in node.iterchildren():
                child.attrib['parentNo'] = str(i)
        for algorithm in self.algorithms:
            algorithm.process_html_part(html, self.body)

    def check(self):
        if self.html == None:
            raise Exception("Please feed a html string first (by calling \"process_html\" method).")

    def predict_node_label(self):
        self.labels = [algorithm.predict_node_label() for algorithm in self.algorithms]
        min_len = min([len(single_ret) for single_ret in self.labels])
        self.labels = [single_ret[:min_len] for single_ret in self.labels]
        self.combined_label = (sum(self.labels) + np.random.choice([1e-7, -1e-7], min_len))/len(self.labels) > 0.5
        return self.combined_label

    def predict_node_prob(self, merge='avg'):
        self.probs = [algorithm.predict_node_prob() for algorithm in self.algorithms]
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
            ret = Preprocess.filter_node_tokens(self.body, self.combined_label, 1)

        else:
            self.predict_node_prob(merge)
            ret = Preprocess.filter_node_tokens(self.body, self.combined_prob, 0.5)
        return ret

    def filter_content_all(self):
        ret = {}
        for idx, pred in enumerate(self.labels):
            ret_singe = Preprocess.filter_node_tokens(self.body, pred, 0.5)
            ret[self.algorithms[idx].__class__.__name__] = ret_singe
        if len(self.algorithms) > 1:
            ret['line_combined' + '-vote'] = self.filter_content('vote')
            ret['line_combined' + '-avg'] = self.filter_content('avg')
            ret['line_combined' + '-max'] = self.filter_content('max')

        return ret


        