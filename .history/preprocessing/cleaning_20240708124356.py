import pandas as pd 
import numpy as np 
data = pd.read_json("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/Preprocessing-Visualization/data/final_dataset.json")
df = pd.DataFrame(data)
data_csv = df.to_csv(data)

df['Price'].describe()