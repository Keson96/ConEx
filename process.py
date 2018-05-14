from tidylib import tidy_document
from lxml.html.clean import Cleaner
import lxml.html
import lxml.etree
import re

class Preprocess(object):
    """
    Preprocess html for later uses
    """
    # cleaner that cleans the whole html document
    cleaner = Cleaner(page_structure=False, links=False, style= True, scripts=True, 
                    kill_tags=['noscript', 'br', 'img', 'code'])
    # cleaner that only cleans <body> tag fragment in html document
    cleaner_body = Cleaner(page_structure=True, links=False, style= True, scripts=True, 
                    kill_tags=['noscript', 'br', 'img', 'code'])
    
    @classmethod
    def clean(cls, html, tidy=True, body_only=False):
        """
        clean html document
        """
        if body_only:
            cleaner = cls.cleaner_body
        else:
            cleaner = cls.cleaner
        if tidy:
            document, errors = tidy_document(html)
            cleaned = cleaner.clean_html(document)
        else:
            cleaned = cleaner.clean_html(html)
        return cleaned
    
    @classmethod
    def get_html_lines(cls, html, tidy=True, body_only=False):
        """
        process html document and split it into lines
        """
        cleaned = cls.clean(html, tidy, body_only)
        html_lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        return html_lines
    
    @classmethod
    def get_html_nodes(cls, html, tidy=True, body_only=False):
        """
        parse html document into a DOM tree, return the root node
        """
        if body_only:
            root = lxml.html.fragment_fromstring(cls.clean(html, tidy, True),  create_parent=True)
        else:
            root = lxml.html.fromstring(cls.clean(html, tidy, False))
        return root
    
    @classmethod
    def get_tokens_by_line(cls, html, tidy=True, body_only=False):
        """
        process html document and split it into lists, each list contains tokens in a line
        """
        html_lines = cls.get_html_lines(html, tidy, body_only)
        tokens = []
        token = ""
        for line in html_lines:
            line_tokens = []
            openTag = False
            token = ""
            for ch in line:
                if ch == '<' and openTag == False:
                    openTag = True
                    if token.strip():
                        line_tokens.extend(token.split())
                    token = "" + ch
                elif ch == '>' and openTag == True:
                    openTag = False
                    token += ch
                    line_tokens.append(token)
                    token = ""
                else:
                    token += ch
            if token.strip():
                line_tokens.extend(token.split())
            tokens.append(line_tokens)
        return tokens
    
    @classmethod
    def get_all_tokens(cls, html, tidy=True, body_only=False):
        """
        process html document and get all tokens
        """
        cleaned = cls.clean(html, tidy, body_only)
        all_tokens = []
        token = ""
        openTag = False
        for ch in cleaned:
            if ch == '<' and openTag == False:
                openTag = True
                if token.strip():
                    all_tokens.extend(token.split())
                token = "" + ch
            elif ch == '>' and openTag == True:
                openTag = False
                token += ch
                all_tokens.append(token)
                token = ""
            else:
                token += ch
        if token.strip():
            all_tokens.append(token)
        return all_tokens

    @classmethod
    def filter_node_tokens(cls, node, pred, threshold = 1):
        """
        filter content tokens given a DOM tree root and a predicted sequence
        """
        tokens = []
        if pred[int(node.attrib['No'])] >= threshold and node.text and node.text.strip():
            tokens.extend(node.text.strip().split())
        for child in node.iterchildren():
            tokens.extend(cls.filter_node_tokens(child, pred))
        if int(node.attrib['No']):
            if pred[int(node.attrib['parentNo'])] >= threshold and node.tail and node.tail.strip():
                tokens.extend(node.tail.strip().split())
        return tokens

class Evaluate(object):
    """
    Evaluate the result of content extraction
    """
    @classmethod
    def lcs_length(cls, a, b):
        """
        compute common sequence length using longest common sequence algorithm
        """
        table = [[0] * (len(b) + 1) for _ in xrange(len(a) + 1)]
        for i, ca in enumerate(a, 1):
            for j, cb in enumerate(b, 1):
                table[i][j] = (
                    table[i - 1][j - 1] + 1 if ca == cb else
                    max(table[i][j - 1], table[i - 1][j]))
        return table[-1][-1]

    @classmethod
    def eval_LCS(cls, pred_seq, gold_seq):
        """
        use longest common sequence algorithm for evaluation
        """
        lcs = cls.lcs_length(pred_seq, gold_seq)
        precision = lcs * 1.0 / len(pred_seq)
        recall = lcs * 1.0 / len(gold_seq)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    @classmethod
    def eval_wordset(cls, pred_seq, gold_seq):
        """
        use wordset method for evaluation
        """
        pred_set, gold_set = set(pred_seq), set(gold_seq)
        intersection = pred_set.intersection(gold_set)
        precision = len(intersection) * 1.0 / len(pred_set)
        recall = len(intersection) * 1.0 / len(gold_set)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    @classmethod
    def eval(cls, pred_seq, gold_seq):
        """
        evaluation scheme used in the experiments
        """
        if len(pred_seq) == 0:
            return 0, 0, 0
        if len(gold_seq) > 30000:
            return cls.eval_wordset(pred_seq, gold_seq)
        else:
            return cls.eval_LCS(pred_seq, gold_seq)
        