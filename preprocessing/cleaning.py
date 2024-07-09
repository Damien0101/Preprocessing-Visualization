import pandas as pd 
import numpy as np 
data = pd.read_json("data/final_dataset.json")
df = pd.DataFrame(data)
data_csv = df.to_csv('data/dataset.csv')

df.drop(columns=['Url'])
print(df.head())

file_postcodes = "preprocessing/georef-belgium-postal-codes.json"
df_postcodes = pd.read_json(file_postcodes)
df_postcodes = df_postcodes[["postcode"]]

# merging properties dataset with postcode df and drop duplicated values
df = df.merge(df_postcodes, how="inner", right_on="postcode", left_on="PostalCode").drop_duplicates()
del df["postcode"]

# filling null values for SwimmingPool by False, since no mention of swimming pool is interpreted as absence of swimming pool
df["SwimmingPool"] = df["SwimmingPool"].fillna(False)

# suggestion to delete the openfire column, because of the absence of True values
del df["Openfire"]

