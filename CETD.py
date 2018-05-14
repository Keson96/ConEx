from process import Preprocess

import numpy as np
import seaborn as sns

class CETD(object):
    """docstring for CETD"""
    def __init__(self, method='CTD'):
        super(CETD, self).__init__()
        self.html = None
        self.body = None
        self.method = method
        self.TD = None
        self.CTD = None
        self.TD_threshold = 0
        self.CTD_threshold = 0

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
        self.compute_statistics(self.body)
        self.TD = [float(elem.attrib['TD']) for elem in self.body.iterdescendants()]
        self.compute_CTD(self.body)
        self.CTD = [float(elem.attrib['CTD']) for elem in self.body.iterdescendants()]

    def process_html_part(self, html, body):
        self.html = html
        self.body = body
        self.compute_statistics(self.body)
        self.TD = [float(elem.attrib['TD']) for elem in self.body.iterdescendants()]
        self.compute_CTD(self.body)
        self.CTD = [float(elem.attrib['CTD']) for elem in self.body.iterdescendants()]

    def check(self):
        if self.html == None:
            raise Exception("Please feed a html string first (by calling \"process_html\" method).")

    def compute_statistics(self, node):
        self.check()
        for child in node.getchildren():
            self.compute_statistics(child)
        if node.text == None:
            CountText = 0
        else:
            CountText = len(node.text.strip())
        if node.tag == 'a':
            CountLinkChar = len(node.text.strip()) if node.text else 0
        else:
            CountLinkChar = 0
        node.attrib['CountChar'] = str(CountText + sum(int(x.attrib['CountChar']) for x in node.getchildren()))
        node.attrib['CountTag'] = str(sum(int(x.attrib['CountTag']) for x in node.getchildren()) + len(node.getchildren()) + 1)
        node.attrib['CountLinkChar'] = str(CountLinkChar + sum(int(x.attrib['CountLinkChar']) for x in node.getchildren()))
        node.attrib['CountLinkTag'] = str(sum(int(x.attrib['CountLinkTag']) for x in node.getchildren()) + 
                                          sum(child.tag =='a' for child in node.getchildren()))
        node.attrib['TD'] = str(float(node.attrib['CountChar'])/max(int(node.attrib['CountTag']),1))
        node.attrib['TD_Sum'] = str(sum(float(child.attrib['TD']) for child in node.getchildren()))


    def compute_CTD(self, body):
        first_part = lambda elem: float(elem.attrib['CountChar']) / max(float(elem.attrib['CountTag']), 1)
        base = lambda elem: np.log(float(elem.attrib['CountChar']) * float(elem.attrib['CountLinkChar']) / 
                                      max(1, (float(elem.attrib['CountChar']) - float(elem.attrib['CountLinkChar'])))
                                  +
                                   float(body.attrib['CountLinkChar']) * float(elem.attrib['CountChar']) / 
                                      max(1, float(body.attrib['CountChar']))
                                  +
                                   np.e)
        number = lambda elem: (float(elem.attrib['CountChar']) * float(elem.attrib['CountTag']) / 
                              max(1, float(elem.attrib['CountLinkChar'])) / max(1, float(elem.attrib['CountLinkTag'])))
        
        for elem in body.iter():
            CTD_elem = 0
            if first_part(elem) != 0 and number(elem) != 0 and base(elem) != 1:
                CTD_elem = first_part(elem)*np.log(number(elem))/np.log(base(elem))
            elem.attrib['CTD'] = str(CTD_elem)
        for elem in body.iter():
            elem.attrib['CTD_Sum'] = str(sum(float(child.attrib['CTD']) for child in elem.getchildren()))


    def find_threshold(self, body):
        tree = body.getroottree()
        TD_elem = max([(elem, float(elem.attrib['TD_Sum'])) for elem in body.iter()], key=lambda x:x[1])[0]
        TD_path_lst = tree.getelementpath(TD_elem).split('/')
        self.TD_threshold = min([float(tree.xpath("/".join(TD_path_lst[:i+1]))[0].attrib['TD']) 
                                for i in range(len(TD_path_lst))])
        CTD_elem = max([(elem, float(elem.attrib['CTD_Sum'])) for elem in body.iter()], key=lambda x:x[1])[0]
        CTD_path_lst = tree.getelementpath(CTD_elem).split('/')
        self.CTD_threshold = min([float(tree.xpath("/".join(CTD_path_lst[:i+1]))[0].attrib['CTD']) 
                                for i in range(len(CTD_path_lst))])

    def predict_node_label(self):
        if np.count_nonzero(self.CTD) == 0:
            self.method = 'TD'
        self.find_threshold(self.body)
        if self.method == 'CTD':
            node_label = [1 if float(elem.attrib["CTD"]) > self.CTD_threshold else 0 for elem in self.body.iter()]
        elif self.method == 'TD':
            node_label = [1 if float(elem.attrib["TD"]) > self.TD_threshold else 0 for elem in self.body.iter()]
        else:
            raise Exception("Wrong method argument: " + self.method)
        return np.array(node_label)

    def predict_node_prob(self, scale=1):
        if self.CTD_threshold == 0:
            prob = np.ones(len(list(self.body.iter())))
        else:
            prob = np.array([1/(1+np.exp(scale*(1 - float(elem.attrib["CTD"])/self.CTD_threshold)))
                            for elem in self.body.iter()])
        return prob

    def filter_content(self):
        pred = self.predict_node_label()
        content = Preprocess.filter_node_tokens(self.body, pred, 1)
        return content



    def plot_TD(self, figsize=(40,10)):
        sns.plt.figure(figsize=figsize)
        sns.plt.bar(np.arange(len(self.TD)), self.TD)
        sns.plt.show()


    def plot_CTD(self, figsize=(40,10)):
        sns.plt.figure(figsize=figsize)
        sns.plt.bar(np.arange(len(self.CTD)), self.CTD)
        sns.plt.show()