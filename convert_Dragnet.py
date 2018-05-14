import sys
import os
import errno
import lxml.html

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
        print("Please use:\n\tpython convert_Dragnet.py DragnetDataPath TargetPath")
        print("------------------------------------")
    Dragnet_path = sys.argv[1]
    Target_path = sys.argv[2]

    for filename in os.listdir(Dragnet_path + 'HTML/'):
        if filename.endswith('.html'):
            with open(Dragnet_path + 'HTML/' + filename) as f:
                try:
                    html = lxml.html.tostring(lxml.html.fromstring(f.read()), encoding='utf-8')
                except:
                    print("-----" + filename)
            html_path = Target_path + 'HTML/' + filename
            checkpath(html_path)
            with open(html_path, 'w') as f:
                f.writelines(html)

    for filename in os.listdir(Dragnet_path + 'Corrected/'):
        if filename.endswith('.txt'):
            with open(Dragnet_path + 'Corrected/' + filename) as f:
                content = f.read().replace('!@#$%^&*()  COMMENTS', '')
            content_path = Target_path + 'Corrected/' + filename
            checkpath(content_path)
            with open(content_path, 'w') as f:
                f.writelines(content)


if __name__ == "__main__":
    main()