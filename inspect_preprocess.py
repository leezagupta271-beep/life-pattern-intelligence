import json
path = r'C:\Users\LENOVO\OneDrive\Documents\Desktop\Life_Pattern_Intelligence\data_cleaning.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Exercise' in src or 'LabelEncoder' in src or 'OneHot' in src or 'fit_transform' in src or 'StandardScaler' in src or 'preprocess' in src:
        print('CELL', i)
        print(src)
        print('---')
