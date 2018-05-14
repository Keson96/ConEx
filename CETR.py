from process import Preprocess
from kmeans import KMeans

from scipy.ndimage.filters import gaussian_filter1d
from sklearn.preprocessing import normalize
import sklearn.cluster
import scipy.spatial
import seaborn as sns
import numpy as np
import itertools

class CETR(object):
    """docstring for CETR"""


    def __init__(self, sigma=1, alpha=3, method='1D'):
        super(CETR, self).__init__()
        self.html = None
        self.method = method
        self.tokens_by_line = None
        self.tagRatio = None
        self.smoothed_TR = None
        self.smoothed_G = None
        self.sigma = sigma
        self.alpha = alpha
        self.distance1D = None
        self.distance2D = None
        self.label1d = None

    def process_html(self, html, tidy=True):
        self.html = html
        self.tokens_by_line = Preprocess.get_tokens_by_line(html, tidy)
        self.compute_tagRatio(html)
        self.smoothed_TR = self.smooth(self.tagRatio)
        self.compute_smoothed_G()

    def process_html_part(self, html, tokens_by_line):
        self.html = html
        self.tokens_by_line = tokens_by_line
        self.compute_tagRatio(html)
        self.smoothed_TR = self.smooth(self.tagRatio)
        self.compute_smoothed_G()

    def check(self):
        if self.html == None:
            raise Exception("Please feed a html string first (by calling \"process_html\" method).")

    def compute_tagRatio(self, html):
        self.check()
        tokens_count = [[0 if token.startswith('<') and token.endswith('>') else len(token) for token in line] 
                for line in self.tokens_by_line]
        self.tagRatio = [sum(line)*1.0/max(1, sum([x == 0 for x in line])) for line in tokens_count]

    def smooth(self, sequence):
        self.check()
        smoothed = gaussian_filter1d(sequence, self.sigma)
        return smoothed

    def compute_smoothed_G(self):
        self.check()
        G = [sum(self.smoothed_TR[i:i + self.alpha])*1.0/self.alpha - self.smoothed_TR[i] 
                    for i in range(0, len(self.smoothed_TR)-self.alpha)]
        self.smoothed_G = abs(self.smooth(G))

    def remove_tag(self, lst):
        filtered = [token for token in lst if not (token.startswith('<') and token.endswith('>'))]
        return filtered

    def predict_line_label(self):
        self.check()
        if self.method == '1D':
            kmeans = sklearn.cluster.KMeans(n_clusters=3)
            pred = kmeans.fit_predict(np.array(self.smoothed_TR).reshape(-1,1))
            self.distance1D = scipy.spatial.distance.cdist(np.array(self.smoothed_TR).reshape(-1,1), kmeans.cluster_centers_)
            self.label1d = np.argmin(kmeans.cluster_centers_)
            line_label = pred != self.label1d
        elif self.method == '2D':
            kmeans = KMeans(n_clusters=3, fixed_centroids={0:[0,0]})
            norm_TR = np.linalg.norm(self.smoothed_TR[:len(self.smoothed_G)])
            norm_G = np.linalg.norm(self.smoothed_G)
            X = np.dstack((self.smoothed_TR[:len(self.smoothed_G)], self.smoothed_G * norm_TR / norm_G))[0]
            pred = kmeans.fit_predict(X)
            self.distance2D = kmeans.distance
            line_label = np.hstack([pred != 0, [False] * self.alpha])
        else:
            raise Exception("Wrong method argument: " + self.method)
        return line_label

    def filter_content(self):
        self.check()
        pred = self.predict_line_label()
        ret = [token for line in itertools.compress(self.tokens_by_line, pred) for token in line
                if (not (token.startswith('<') and token.endswith('>')) and token.strip())]
        return ret

    def predict_line_prob(self, scale = 5):
        if self.method == '1D':
            prob = 1 - normalize(np.exp(-scale*normalize(self.distance1D, axis=1, norm='l2')), axis=1, norm='l1')[:,self.label1d]
        elif self.method == '2D':
            prob = 1 - normalize(np.exp(-scale*normalize(self.distance2D, axis=1, norm='l2')), axis=1, norm='l1')[:,0]
            prob = np.hstack([prob, [0] * self.alpha])
        else:
            raise Exception("Wrong method argument: " + self.method)
        return prob

    def plot_tagRatio(self, figure_size=(40,10)):
        self.check()
        sns.plt.figure(figsize=figure_size)
        sns.plt.bar(np.arange(len(self.tagRatio)), self.tagRatio)
        sns.plt.show()

    def plot_smoothed_G(self, figure_size=(40,10)):
        self.check()
        sns.plt.figure(figsize=figure_size)
        sns.plt.bar(np.arange(len(self.smoothed_G)), self.smoothed_G)
        sns.plt.show()
