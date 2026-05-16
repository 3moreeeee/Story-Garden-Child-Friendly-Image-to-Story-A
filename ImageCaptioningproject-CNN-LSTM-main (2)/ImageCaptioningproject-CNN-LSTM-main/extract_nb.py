import json
import sys

def extract_code(ipynb_path, py_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(py_path, 'w', encoding='utf-8') as f:
        for cell in nb.get('cells', []):
            if cell['cell_type'] == 'code':
                source = ''.join(cell.get('source', []))
                f.write(source + '\n\n')

if __name__ == '__main__':
    extract_code('imgCaptioningprojectRayan.ipynb', 'extracted_notebook.py')
