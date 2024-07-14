import pandas as pd

final_df = pd.read_csv("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/Preprocessing-Visualization/data/final_set.csv")

df_impute = final_df.select_dtypes(include=['int', 'float'])

print(df_impute.dtypes)
