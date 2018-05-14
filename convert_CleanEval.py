import sys
import os
import errno

def process(content_lines):
    return [line[line.find('>') + 1:] if '<' in line and '>' in line else line for line in content_lines]
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
        print("Please use:\n\tpython convert_CleanEval.py CleanEvalDataPath TargetPath")
        print("------------------------------------")
    CleanEval_path = sys.argv[1]
    Target_path = sys.argv[2]

    zh_original = []
    zh_cleaned = []
    en_original = []
    en_cleaned = []
    final_input = []
    final_gold = []
    for filename in os.listdir(CleanEval_path + 'zh-original/'):
        zh_original.append(filename.split('.')[0])
    for filename in os.listdir(CleanEval_path + 'zh-cleaned/'):
        zh_cleaned.append(filename.split('-')[0])
    for filename in os.listdir(CleanEval_path + 'en-original/'):
        en_original.append(filename.split('.')[0])
    for filename in os.listdir(CleanEval_path + 'en-cleaned/'):
        en_cleaned.append(filename.split('-')[0])
    for filename in os.listdir(CleanEval_path + 'finalrun-input/'):
        final_input.append(filename.split('.')[0])
    for filename in os.listdir(CleanEval_path + 'GoldStandard/'):
        final_gold.append(filename.split('.')[0])

    zh = set(zh_original).intersection(set(zh_cleaned))
    en = set(en_original).intersection(set(en_cleaned))
    final = set(final_input).intersection(set(final_gold))

    for idx in zh:
        with open(CleanEval_path + 'zh-original/' + idx + '.html') as f:
            try:
                html = f.read().decode('gbk').encode('utf-8')
            except UnicodeDecodeError:
                html = f.read().decode('gb2312').encode('utf-8')
        html_filename = Target_path + 'zh-original/' + idx + '.html'
        checkpath(html_filename)
        with open(html_filename, 'w') as f:
            html_lines = html.split('\n')
            if '<text id=' in html_lines[0]:
                f.writelines('\n'.join(html_lines[1:]))
            else:
                f.writelines(html)
        with open(CleanEval_path + 'zh-cleaned/' + idx + '-cleaned.txt') as f:
            content = f.read().split('\n')
            if 'URL' in content[0]:
                for line_num, line in enumerate(content):
                    if 'http' in line:
                        break
                content = content[line_num + 1:]
        content_filename = Target_path + 'zh-cleaned/' + idx + '.txt'
        checkpath(content_filename)
        with open(content_filename, 'w') as f:
            for line in process(content):
                f.writelines(line + '\n')

    for idx in en:
        with open(CleanEval_path + 'en-original/' + idx + '.html') as f:
            html = f.read()
        html_filename = Target_path + 'en-original/' + idx + '.html'
        checkpath(html_filename)
        with open(html_filename, 'w') as f:
            html_lines = html.split('\n')
            if '<text id=' in html_lines[0]:
                f.writelines('\n'.join(html_lines[1:]))
            else:
                f.writelines(html)
        with open(CleanEval_path + 'en-cleaned/' + idx + '-cleaned.txt') as f:
            content = f.read().split('\n')
            if 'URL' in content[0]:
                for line_num, line in enumerate(content):
                    if 'http' in line:
                        break
                content = content[line_num + 1:]
        content_filename = Target_path + 'en-cleaned/' + idx + '.txt'
        checkpath(content_filename)
        with open(content_filename, 'w') as f:
            for line in process(content):
                f.writelines(line + '\n')

    for idx in final:
        with open(CleanEval_path + 'finalrun-input/' + idx + '.html') as f:
            html = f.read()
        html_filename = Target_path + 'final-original/' + idx + '.html'
        checkpath(html_filename)
        with open(html_filename, 'w') as f:
            html_lines = html.split('\n')
            if '<text id=' in html_lines[0]:
                f.writelines('\n'.join(html_lines[1:]))
            else:
                f.writelines(html)
        with open(CleanEval_path + 'GoldStandard/' + idx + '.txt') as f:
            content = f.read().split('\n')
            if 'URL' in content[0]:
                for line_num, line in enumerate(content):
                    if 'http' in line:
                        break
                content = content[line_num + 1:]
        content_filename = Target_path + 'final-cleaned/' + idx + '.txt'
        checkpath(content_filename)
        with open(content_filename, 'w') as f:
            for line in process(content):
                f.writelines(line + '\n')


if __name__ == "__main__":
    main()