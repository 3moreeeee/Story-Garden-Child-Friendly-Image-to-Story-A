import json
import io

with open('imgCaptioningprojectRayan.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

source_code = []
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_code.append(f"# --- CELL {idx} ---\n" + "".join(cell.get('source', [])) + "\n")

with open('notebook_source.py', 'w', encoding='utf-8') as f:
    f.write("\n".join(source_code))
print("Extraction complete.")
