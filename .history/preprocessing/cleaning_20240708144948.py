import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
file = "/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/Preprocessing-Visualization/data/final_dataset.json"
df = pd.read_json(file)
# data_csv = df.to_csv(data)

def wrong_postal_code(x):
    wrong_codes = []
    if x < 1000 or x > 10000:
        wrong_codes.append(x)
    return len(wrong_codes    

df["PostalCode"].apply(wrong_postal_code)



