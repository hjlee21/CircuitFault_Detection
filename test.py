import numpy as np
import pandas as pd

file = './Data_PINN/NORM_NORM_1.csv'
df = pd.read_csv(file)
print(df['V10'])