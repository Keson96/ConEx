from process import Preprocess

import sklearn.cluster
from sklearn.preprocessing import normalize
import scipy.spatial
import seaborn as sns
import numpy as np
import itertools

class CTTD(object):
    """docstring for CTTD"""

    def __init__(self, lambda_param=0.5):
        super(CTTD, self).__init__()
        self.html = None
        self.lambda_param = lambda_param
        self.tokens_by_line = None
        self.CTTD = None
        self.smoothed_CTTD = None
        self.distance = None
        self.min_cluster_label = None

    def process_html(self, html, tidy=True):
        self.html = html
        self.tokens_by_line = Preprocess.get_tokens_by_line(html, tidy)
        self.compute_CTTD()
        self.smooth()

    def process_html_part(self, html, tokens_by_line):
        self.html = html
        self.tokens_by_line = tokens_by_line
        self.compute_CTTD()
        self.smooth()


    def check(self):
        if self.html == None:
            raise Exception("Please feed a html string first (by calling \"process_html\" method).")

    def compute_CTTD(self):
        stats = []
        for tokens in self.tokens_by_line:
            stats_line = {"TC":0, "LTC":0, "TG":0, "P":0, "CTTD":0}
            linkText = False
            for token in tokens:
                if token.startswith('<') and token.endswith('>'):
                    if token.startswith('<a'):
                        linkText = True
                    elif token.startswith('</a'):
                        linkText = False
                    elif token.startswith('<p'):
                        stats_line['P'] += 1
                    stats_line['TG'] += 1
                elif linkText == True:
                    stats_line["TC"] += 1
                    stats_line["LTC"] += 1
                else:
                    stats_line["TC"] += 1
            if stats_line['TC'] == 0:
                stats_line['CTTD'] = 0
            else:
                stats_line['CTTD'] = stats_line['TC'] + self.lambda_param * stats_line['LTC'] + stats_line['TG'] - stats_line['P']
            stats.append(stats_line)
        self.CTTD = np.array([line['CTTD'] for line in stats], dtype='float')

    def smooth(self):
        self.smoothed_CTTD = np.array([(self.CTTD[max(0, i-2)] 
                                        + 2*self.CTTD[max(0, i-1)] 
                                        + 4*self.CTTD[i] 
                                        + 2*self.CTTD[min(i+1, self.CTTD.shape[0]-1)] 
                                        + self.CTTD[min(i+2, self.CTTD.shape[0]-1)])/10 
                                for i in range(self.CTTD.shape[0])])

    def filter_content(self):
        pred = self.predict_line_label()
        ret = [token for line in itertools.compress(self.tokens_by_line, pred) for token in line
                if (not (token.startswith('<') and token.endswith('>')) and token.strip())]
        return ret

    def predict_line_label(self):
        kmeans = sklearn.cluster.KMeans(n_clusters=2, n_jobs=4)
        pred = kmeans.fit_predict(self.smoothed_CTTD.reshape(-1, 1))
        self.distance = scipy.spatial.distance.cdist(np.array(self.smoothed_CTTD).reshape(-1,1), kmeans.cluster_centers_)
        self.min_cluster_label = kmeans.cluster_centers_.argmin()
        if self.min_cluster_label == 1:
            pred = np.logical_not(pred).astype(int)
        line_label = pred.astype('bool')
        return line_label

    def predict_line_prob(self, scale=5):
        prob = 1 - normalize(np.exp(-scale*normalize(self.distance, axis=1, norm='l2')), axis=1, norm='l1')[:,self.min_cluster_label]
        return prob

    def plot_CTTD(self, figure_size=(40,10)):
        self.check()
        sns.plt.figure(figsize=figure_size)
        sns.plt.bar(np.arange(len(self.CTTD)), self.CTTD)
        sns.plt.show()

    def plot_smoothed_CTTD(self, figure_size=(40,10)):
        self.check()
        sns.plt.figure(figsize=figure_size)
        sns.plt.bar(np.arange(len(self.smoothed_CTTD)), self.smoothed_CTTD)
        sns.plt.show()