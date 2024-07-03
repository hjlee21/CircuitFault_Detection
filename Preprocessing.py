import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataframeProcessor:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.scaler = MinMaxScaler()

        self.fault_type_mapping = {
            'NORM': 0,
            'FSEN': 1,
            'FAMP': 2,
            'FLPF': 3
        }

    def transform_string(self, input_string):
        trace_part = input_string.split('Trace ')[1]
        trace_number = trace_part.split('::')[0]

        if 'TIME' in input_string:
            return 'TIME'+trace_number
        else:
            return 'V'+trace_number
    
    def call_data(self, file):
        df = pd.read_csv(file)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = [self.transform_string(col) for col in df.columns]
        return df
    
    def remove_duplicate_time_columns(self, df_file):
        self.cols_to_keep = []
        time_count = 0
        for col in df_file.columns:
            if 'TIME' in col:
                if time_count == 0:
                    self.cols_to_keep.append(col)
                time_count += 1
            else:
                self.cols_to_keep.append(col)
        df = df_file[self.cols_to_keep]
        return df
    
    def labeling(self, filename):
        fault_type = filename.split('_')[0]
        if fault_type in self.fault_type_mapping:
            file_path = os.path.join(self.dir_path, filename)
            df = pd.read_csv(file_path)
            fault_label = self.fault_type_mapping[fault_type]
            df['label'] = fault_label
            df.to_csv(file_path, index=False)
            print(f'{filename} lebeled successfully')

dir_path = './Plan_DataAcquisition/data/'
dir_path2= './Plan_DataAcquisition/data_pd/'
dir_path3= './Plan_DataAcquisition/validation'

dir_save = './Plan_DataAcquisition/data_pd'
dir_save2= './Plan_DataAcquisition/data_pd_minmax'

processor = DataframeProcessor(dir_path)

# ================================================================= find scaler #!TODO dict로 업데이트 해보기
min_value, max_value, min_scale, max_scale = [], [], [], []
scaler = MinMaxScaler()
for ifile in os.listdir(dir_path2):
    if ifile.endswith('.csv'):
        file_path = os.path.join(dir_path2, ifile)
        df = pd.read_csv(file_path).iloc[:,1:]
        each_min = df.min(axis=0).values
        each_max = df.max(axis=0).values
        min_value.append(each_min)
        max_value.append(each_max)

min_scale.append(np.array(min_value).min(axis=0))
max_scale.append(np.array(max_value).max(axis=0))
scaler.fit([min_scale[0], max_scale[0]])
# ================================================================= data processing 1

processor = DataframeProcessor(dir_path)

for filename in os.listdir(dir_path):
    file_path = os.path.join(dir_path, filename)
    df = processor.call_data(file_path)
    df = processor.remove_duplicate_time_columns(df)

    save_path = os.path.join(dir_save, filename)
    df.to_csv(dir_save, index=False)
    print(f'{filename} processed and saved to : {save_path}')

# ================================================================= data processing 2: minmax norm
for filename in os.listdir(dir_path2):
    file_path = os.path.join(dir_path2, filename)
    df = pd.read_csv(file_path)
    df_values = df.iloc[:, 1:].values
    df_scaled = scaler.transform(df_values)
    df_ = pd.DataFrame(df_scaled, columns=df.columns[1:])

    Time_column = df.iloc[:, 0]
    df_scaled = pd.concat([Time_column, df_], axis = 1)

    save_path2 = os.path.join(dir_save2, filename)
    df_scaled.to_csv(save_path2, index=False)
    print(f'{filename} processed and saved to : {save_path2}')

# ================================================================= data processing 3: Labeling
processor = DataframeProcessor(dir_path3)
for filename in os.listdir(dir_path3):
    file_path = os.path.join(dir_path3, filename)
    df = processor.labeling(filename)


