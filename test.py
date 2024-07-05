import os
import pandas as pd

def get_label_from_filename(filename):
    if 'FAMP' in filename:   return 0  # 고장 라벨 0
    elif 'FLPF' in filename: return 1  
    elif 'FLPF' in filename: return 2  
    elif 'FSEN' in filename: return 3
    elif 'NF' in filename:   return 4
    else: raise ValueError(f"Unknown label for file: {filename}")

folder = './Data/data_training'
filename = 'FAMP_C1D_0.1.csv'

file_path = os.path.join(folder, filename)
df = pd.read_csv(file_path)
# label = get_label_from_filename(filename)
label = df['label']

print(label)

