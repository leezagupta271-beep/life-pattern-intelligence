import json
path = r'C:\Users\LENOVO\OneDrive\Documents\Desktop\Life_Pattern_Intelligence\data_cleaning.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'dd' in src or 'r2_score' in src or 'y_pred' in src or 'predict' in src:
        print('CELL', i)
        print(src)
        print('---')
