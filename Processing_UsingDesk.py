import os
import time
import pandas as pd

data_folder = './Data/data_training'
input_cols = ['V2', 'V8', 'V10']
sequence_length = 10

data = []
labels = []

def get_label_from_filename(filename):
    if 'NORM' in filename:   return 0  # 고장 라벨 0
    elif 'FSEN' in filename: return 1  
    elif 'FAMP' in filename: return 2  
    elif 'FLPF' in filename: return 3
    else: raise ValueError(f"Unknown label for file: {filename}")
    
start_time = time.time()

for filename in os.listdir(data_folder)*20:
    if filename.endswith('.csv'):
        file_path = os.path.join(data_folder, filename)
        df = pd.read_csv(file_path)
        label = get_label_from_filename(filename)
        df_cols = df[input_cols].values
        for i in range(len(df) - sequence_length):
            data.append(df_cols[i:i+sequence_length])
            labels.append([label])
            
run_time = time.time() - start_time
print(f'Time {run_time}')
print(f'len(data) : {len(data)}, len(labels) : {len(labels)}')


# ===========
from joblib import Parallel, delayed

def process_file(file_path, input_cols, sequence_length, label):
    df = pd.read_csv(file_path)
    df_cols = df[input_cols].values
    data = []
    labels = []
    for i in range(len(df) - sequence_length):
        data.append(df_cols[i:i+sequence_length])
        labels.append([label])
    return data, labels

def process_data_folder(data_folder, input_cols, sequence_length):
    data = []
    labels = []

    # 병렬로 파일을 처리
    results = Parallel(n_jobs=-1)(delayed(process_file)(
        os.path.join(data_folder, filename), 
        input_cols, 
        sequence_length, 
        get_label_from_filename(filename)
    ) for filename in os.listdir(data_folder)*20 if filename.endswith('.csv'))
    
    # 결과를 합치기
    for file_data, file_labels in results:
        data.extend(file_data)
        labels.extend(file_labels)

    return data, labels

start_time = time.time()

data, labels = process_data_folder(data_folder, input_cols, sequence_length)

run_time = time.time() - start_time
print(f'Time {run_time}')
print(f'len(data) : {len(data)}, len(labels) : {len(labels)}')

# 결론 _ 더 느림.