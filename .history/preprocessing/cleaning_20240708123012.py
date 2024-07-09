import pandas as pd 
import numpy as np 
data = pd.read_json("data/final_dataset.json")
df = pd.DataFrame(data)
data_csv = df.to_csv('data/dataset.csv')
