from process import Preprocess

import numpy as np
import seaborn as sns
import itertools

class BTE(object):
    """
    Body Text Extracting Algorithm
    """
    def __init__(self, prior=0.7):
        super(BTE, self).__init__()
        self.html = None
        self.tokens = None
        self.labels = None  # 01 sequence indicating whether a token is word or tag
        self.cumsum = None  # compute cumulative sum first for the optimization process
        self.min_i, self.min_j = None, None  # left and right indices of the content area
        self.prior = prior  # posibility assigned to the word tokens in the content area

    def check(self):
        if self.html == None:
            raise Exception("Please feed a html string first (by calling \"process_html\" method).")

    def process_html(self, html, tidy=True, body_only=False):
        """
        process html and compute some necessary statistics
        """
        self.html = html
        self.tokens = Preprocess.get_all_tokens(self.html, tidy, body_only)
        self.labels = np.array([1 if token.startswith('<') and token.endswith('>') else 0
                                    for token in self.tokens])
        self.cumsum = np.cumsum(self.labels)

    def process_html_part(self, html, tokens):
        """
        process html and compute some necessary statistics when BTE is a component in the combined system
        use tokens outputed by the caller rather than processing html again
        """
        self.html = html
        self.tokens = tokens
        self.labels = np.array([1 if token.startswith('<') and token.endswith('>') else 0
                                    for token in self.tokens])
        self.cumsum = np.cumsum(self.labels)

    def predict_token_label(self):
        """
        predict the labels for all tokens, 1 indicating content, 0 indicating noise
        """
        self.check()
        self.min_i, self.min_j = 0, 0
        minObj = float('-inf')
        for i in range(1, len(self.cumsum)-1):
            for j in range(i+1, len(self.cumsum)):
                T = self.cumsum[i-1] + ((j - i + 1) - (self.cumsum[j] - self.cumsum[i-1])) + (self.cumsum[-1] - self.cumsum[j])
                if T > minObj:
                    self.min_i, self.min_j = i, j
                    minObj = T
        token_label = np.array([0 if i > self.min_j or i < self.min_i else 1
                            for i, token in enumerate(self.tokens)
                                if not (token.startswith('<') and token.endswith('>'))])
        return token_label

    def predict_token_prob(self):
        """
        predict the probabilities for all tokens
        """
        prob = np.array([self.prior if self.min_i <= i <= self.min_j else 1 - self.prior
                            for i, token in enumerate(self.tokens) 
                                if not (token.startswith('<') and token.endswith('>'))])
        return prob


    def filter_content(self):
        """
        use predicted labels to filter tokens, return a list of content tokens
        """
        pred = self.predict_token_label()
        ret = list(itertools.compress(
                        filter(lambda token:not(token.startswith('<') and token.endswith('>')),self.tokens),
                    pred))
        return ret

    def plot(self):
        """
        plot cumulative sum
        """
        sns.plt.plot(self.cumsum)
        sns.plt.show()