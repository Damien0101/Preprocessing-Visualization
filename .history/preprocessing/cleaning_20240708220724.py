import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
file = "/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/Preprocessing-Visualization/data/final_dataset.json"
df = pd.read_json(file)
# data_csv = df.to_csv(data)

# # importing json with postcodes, and reducing the postcode df to the postcode column
# file_postcodes = "preprocessing/georef-belgium-postal-codes.json"
# df_postcodes = pd.read_json(file_postcodes)
# df_postcodes = df_postcodes[["postcode"]]

# # merging properties dataset with postcode df and drop duplicated values
# df = df.merge(df_postcodes, how="inner", right_on="postcode", left_on="PostalCode").drop_duplicates()
# del df["postcode"]

# # filling null values for SwimmingPool by False, since no mention of swimming pool is interpreted as absence of swimming pool
# df["SwimmingPool"] = df["SwimmingPool"].fillna(False)

# # suggestion to delete the openfire column, because of the absence of True values
# del df["Openfire"]

df = df[[""]]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Create a ColumnTransformer to handle the different types of data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create a pipeline that first transforms the data and then applies the iterative imputer
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', IterativeImputer())
])

# Fit and transform the data
df_imputed = pipeline.fit_transform(df)

# Convert the result back to a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=pipeline.named_steps['preprocessor'].get_feature_names_out())
print("\nDataFrame after MICE imputation and one-hot encoding:")
print(df_imputed.head(50))

