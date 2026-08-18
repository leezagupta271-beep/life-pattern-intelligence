import json
path = r'C:\Users\LENOVO\OneDrive\Documents\Desktop\Life_Pattern_Intelligence\data_cleaning.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'RandomForestRegressor' in src or 'train_test_split' in src or 'Life' in src or 'target' in src or 'X =' in src or 'y =' in src:
        print('CELL', i)
        print(src)
        print('---')
