from ConEx_line  import ConEx_line
from ConEx_token import ConEx_token
from ConEx_dom   import ConEx_dom
from process import Preprocess

import re
import numpy as np
import itertools
import lxml.html

class ConEx(object):
    """docstring for ConEx"""
    def __init__(self, line_algos, token_algos, dom_algos):
        super(ConEx, self).__init__()
        self.html = None
        self.method = '1-step'
        self.conex_line  = ConEx_line(line_algos)
        self.conex_token = ConEx_token(token_algos)
        self.conex_dom   = ConEx_dom(dom_algos)

    def process_html(self, html, tidy=False):
        self.html = html
        self.html = self.html.replace('&nbsp;', ' ')
        self.html = self.html.replace('&amp;', '&')
        self.html = self.html.replace('&#10;', '')
        self.html = self.html.replace('\r', ' ')
        for m in re.finditer('<[^<>]*[\n]+[^<>]*>', self.html):
            start_pos, end_pos = m.span()
            self.html = self.html[:start_pos] + self.html[start_pos:end_pos].replace('\n',' ') + self.html[end_pos:]
        self.html = lxml.html.tostring(lxml.html.fromstring(self.html), pretty_print=True).decode('iso-8859-1')
        m = re.search(r"<body.*</body>", Preprocess.clean(self.html, tidy), re.DOTALL)
        html_body = ""
        if m:
            html_body = m.group()
        else:
            raise Exception("No body tag in this HTML document")
        
        self.conex_token.process_html(html_body, tidy=tidy, body_only=True)
        self.conex_line.process_html(html_body, tidy=tidy, body_only=True)
        self.conex_dom.process_html(html_body, tidy=tidy, body_only=True)

    def run_algorithms(self):
        self.conex_line.predict_line_label()
        self.conex_token.predict_token_label()
        self.conex_dom.predict_node_label()
        self.conex_line.predict_line_prob()
        self.conex_token.predict_token_prob()
        self.conex_dom.predict_node_prob()

    def predict_label(self):
        if self.method == '1-step':
            line_labels  = self.conex_line.labels
            token_labels = self.conex_token.labels
            dom_labels   = self.conex_dom.labels
            # convert line -> token
            line_labels_token = [
                [labels[idx] 
                    for idx, line in enumerate(self.conex_line.tokens_by_line)
                        for token in line if not (token.startswith('<') and token.endswith('>')) and token.strip()]
                for labels in line_labels]
            # convert dom -> token
            dom_labels_token = []
            for labels in self.conex_dom.labels:
                labels_token = []
                for idx, node in enumerate(self.conex_dom.body.iter()):
                    if node.text and node.text.strip():
                        for token in node.text.strip().split():
                            labels_token.append(labels[idx])
                    if node.tail and node.tail.strip():
                        for token in node.tail.strip().split():
                            labels_token.append(labels[idx])
                dom_labels_token.append(labels_token)
            #print(np.array(token_labels).shape, np.array(line_labels_token).shape, np.array(dom_labels_token).shape)
            # ensemble
            combined_label = (np.random.choice([1e-7, -1e-7], np.array(token_labels).shape[1]) + 
                                np.average(np.vstack([
                                            np.array(token_labels), 
                                            np.array(line_labels_token),
                                            np.array(dom_labels_token)]), axis=0)
                            ) > 0.5

        elif self.method == '2-steps':
            line_label  = self.conex_line.combined_label
            dom_label   = self.conex_dom.combined_label
            token_label = self.conex_token.combined_label
            # convert line -> token
            line_label_token = [line_label[idx]
                    for idx, line in enumerate(self.conex_line.tokens_by_line)
                        for token in line if not (token.startswith('<') and token.endswith('>')) and token.strip()]
                
            # convert dom -> token
            dom_label_token = []
            for idx, node in enumerate(self.conex_dom.body.iter()):
                if node.text and node.text.strip():
                    for token in node.text.strip().split():
                        dom_label_token.append(dom_label[idx])
                if node.tail and node.tail.strip():
                    for token in node.tail.strip().split():
                        dom_label_token.append(dom_label[idx])
            # ensemble
            combined_label = (np.random.choice([1e-7, -1e-7], len(token_label)) + 
                                np.average([np.array(token_label),
                                            np.array(line_label_token),
                                            np.array(dom_label_token)], axis=0)
                            ) > 0.5
        else:
            raise Exception('Wrong method argument: ' + self.method)

        return combined_label

    def predict_prob(self, merge='avg'):
        if self.method == '1-step':
            line_probs  = self.conex_line.probs
            token_probs = self.conex_token.probs
            dom_probs   = self.conex_dom.probs
            # convert line -> token
            line_probs_token = [
                [probs[idx] 
                    for idx, line in enumerate(self.conex_line.tokens_by_line)
                        for token in line if not (token.startswith('<') and token.endswith('>')) and token.strip()]
                for probs in line_probs]
            # convert dom -> token
            dom_probs_token = []
            for probs in self.conex_dom.probs:
                probs_token = []
                for idx, node in enumerate(self.conex_dom.body.iter()):
                    if node.text and node.text.strip():
                        for token in node.text.strip().split():
                            probs_token.append(probs[idx])
                    if node.tail and node.tail.strip():
                        for token in node.tail.strip().split():
                            probs_token.append(probs[idx])
                dom_probs_token.append(probs_token)
            # ensemble
            if merge == 'avg':
                merge_func = np.average
            elif merge == 'max':
                merge_func = np.max
            else:
                raise Exception("Wrong merge argument: " + merge)
            combined_prob = merge_func(np.vstack([
                                            np.array(token_probs), 
                                            np.array(line_probs_token),
                                            np.array(dom_probs_token)]), axis=0)

        elif self.method == '2-steps':
            line_prob  = self.conex_line.combined_prob
            dom_prob   = self.conex_dom.combined_prob
            token_prob = self.conex_token.combined_prob
            # convert line -> token
            line_prob_token = [line_prob[idx]
                    for idx, line in enumerate(self.conex_line.tokens_by_line)
                        for token in line if not (token.startswith('<') and token.endswith('>')) and token.strip()]
                
            # convert dom -> token
            dom_prob_token = []
            for idx, node in enumerate(self.conex_dom.body.iter()):
                if node.text and node.text.strip():
                    for token in node.text.strip().split():
                        dom_prob_token.append(dom_prob[idx])
                if node.tail and node.tail.strip():
                    for token in node.tail.strip().split():
                        dom_prob_token.append(dom_prob[idx])
            # ensemble
            if merge == 'avg':
                merge_func = np.average
            elif merge == 'max':
                merge_func = np.max
            else:
                raise Exception("Wrong merge argument: " + merge)
            combined_prob = merge_func([np.array(token_prob),
                                            np.array(line_prob_token),
                                            np.array(dom_prob_token)], axis=0)
        else:
            raise Exception('Wrong method argument: ' + self.method)

        return combined_prob

    def filter_content(self, merge='vote'):
        if merge == 'vote':
            pred = self.predict_label()
            words = filter(lambda token:not(token.startswith('<') and token.endswith('>')), self.conex_token.tokens)
            ret = list(itertools.compress(words, pred))
        else:
            prob = self.predict_prob(merge)
            words = filter(lambda token:not(token.startswith('<') and token.endswith('>')), self.conex_token.tokens)
            ret = list(itertools.compress(words, prob > 0.5))
        return ret

    def filter_content_all(self):
        ret_all = {}
        ret_line =  self.conex_line.filter_content_all()
        ret_token = self.conex_token.filter_content_all()
        ret_dom = self.conex_dom.filter_content_all()
        ret_all.update(ret_line)
        ret_all.update(ret_token)
        ret_all.update(ret_dom)
        self.method = '1-step'
        ret_all['all-vote-' + self.method] = self.filter_content('vote')
        ret_all['all-avg-' + self.method] = self.filter_content('avg')
        ret_all['all-max-' + self.method] = self.filter_content('max')
        self.method = '2-steps'
        ret_all['all-vote-' + self.method] = self.filter_content('vote')
        ret_all['all-avg-' + self.method] = self.filter_content('avg')
        ret_all['all-max-' + self.method] = self.filter_content('max')
        return ret_all
        


        