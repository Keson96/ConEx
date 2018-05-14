from process import Preprocess

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import itertools
from collections import OrderedDict
from sklearn.cluster import KMeans

class CCB(object):
    """
    Content Code Blurring Algorithm
    """
    def __init__(self, sigma=30):
        super(CCB, self).__init__()
        self.html = None
        self.tokens = None
        self.ATCCV = None  # token-level content code vector that ignores <a> tag
        self.ATCCB = None  # blurred ATCCV
        self.sigma = sigma  # parameter for blurring

    def check(self):
        if self.html == None:
            raise Exception("Please feed a html string first (by calling \"process_html\" method).")

    def process_html(self, html, tidy=True, body_only=False):
        """
        process html and compute some necessary statistics
        """
        self.html = html
        self.tokens = Preprocess.get_all_tokens(html, tidy, body_only)
        self.compute_statistics()
        self.ATCCB = self.blurring(self.ATCCV.values())

    def process_html_part(self, html, tokens):
        """
        process html and compute some necessary statistics when BTE is a component in the combined system
        use tokens outputed by the caller rather than processing html again
        """
        self.html = html
        self.tokens = tokens
        self.compute_statistics()
        self.ATCCB = self.blurring(self.ATCCV.values())


    def compute_statistics(self):
        """
        compute ATCCV
        """
        self.check()
        lst = [(i, 0.0) if token.startswith('<') and token.endswith('>') else (i, 1.0) 
                                    for i, token in enumerate(self.tokens)
                                        if not (token.startswith('<a') or token.startswith('</a'))]
        self.ATCCV = OrderedDict(lst)

    def blurring(self, seq):
        """
        blur sequence using gaussian filter
        """
        return gaussian_filter1d(seq, self.sigma)

    def predict_token_label(self):
        """
        predict the labels for all tokens, 1 indicating content, 0 indicating noise
        """
        token_label = np.array([0 if self.ATCCB[self.ATCCV.keys().index(i)] <= 0.75 else 1
                            for i, token in enumerate(self.tokens) 
                                if not (token.startswith('<') and token.endswith('>'))])
        return token_label

    def predict_token_prob(self, scale=10):
        """
        predict the probabilities for all tokens
        """
        prob = np.array([1/(1+np.exp(scale*(0.75 - self.ATCCB[self.ATCCV.keys().index(i)])))
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


    def plot(self, seq, blurred_seq=[], figsize = (15, 1)):
        """
        plot original sequence and blurred sequence
        """
        x = np.linspace(0,len(seq), len(seq))
        extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
        if blurred_seq != []:
            figsize = (15, 2)
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)
            ax2.imshow(blurred_seq[np.newaxis,:], cmap="binary", aspect="auto", extent=extent)
            ax2.set_yticks([])
        else:
            fig, ax1 = plt.subplots(nrows=1, figsize=figsize)
        ax1.imshow(seq[np.newaxis,:], cmap="binary", aspect="auto", extent=extent)
        ax1.set_yticks([])
        plt.show()