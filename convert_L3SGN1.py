import sys
import os
import errno
import lxml.html
from lxml import etree

def checkpath(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def main():
    if len(sys.argv[1:]) != 2:
    	print("--------Missing Arguments--------")
        print("Please use:\n\tpython convert_L3SGN1.py L3SGN1_DataPath TargetPath")
        print("------------------------------------")
    L3SGN1_path = sys.argv[1]
    Target_path = sys.argv[2]

    for filename in os.listdir(L3SGN1_path + 'original/'):
        if filename.endswith('.html'):

            with open(L3SGN1_path + 'original/' + filename) as f:
                html = f.read()
            with open(L3SGN1_path + 'annotated/' + filename) as f:
                ex_html = f.read()
            
            ex_root = lxml.html.fromstring(ex_html)
            ex_tree = etree.ElementTree(ex_root)
            nodes = [node for node in ex_root.xpath('.//span[starts-with(@class, "x-nc-sel")]') 
                 if not node.attrib['class'].endswith('0')]
            content = '\n'.join([node.text.strip() for node in nodes if node.text])

            html_path = Target_path + 'original/' + filename
            checkpath(html_path)
            with open(html_path, 'w') as f:
                f.writelines(html)

            content_path = Target_path + 'annotated/' + filename[:-5] + '.txt'
            checkpath(content_path)
            with open(content_path, 'w') as f:
                f.writelines(content.encode('utf-8'))

if __name__ == "__main__":
    main()