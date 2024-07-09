import pandas as pd 
import numpy as np 
data = pd.read_json(ù)
df = pd.DataFrame(data)
data_csv = df.to_csv('data/dataset.csv')

df['Price'].describe()