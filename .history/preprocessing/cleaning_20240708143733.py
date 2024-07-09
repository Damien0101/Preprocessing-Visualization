import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
file = "/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/Preprocessing-Visualization/data/final_dataset.json"
df = pd.read_json(file)
# data_csv = df.to_csv(data)

prices = df[df["Price"] == 1]
print
plt.boxplot(prices, notch=None, vert=None, patch_artist=None, widths=None)
plt.show()

