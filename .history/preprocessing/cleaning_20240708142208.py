import pandas as pd 
import numpy as np
import matplotlib 
file = "/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/Preprocessing-Visualization/data/final_dataset.json"
df = pd.read_json(file)
# data_csv = df.to_csv(data)

prices = df["Price"]
matplotlib.pyplot.boxplot(data, notch=None, vert=None, patch_artist=None, widths=None)

