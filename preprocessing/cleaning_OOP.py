import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

class DataCleaning:
    def __init__(self, df):
        self.df = df
    
    def remove_values(self, column, values):
        self.df[column] = self.df[column].apply(lambda x: x if x in values else None)
        
    def remove_outliers(self, column, lower_bound=None, upper_bound=None):
        non_null_values = self.df[column].notna()
        if lower_bound is not None:
            non_null_values &= self.df[column] >= lower_bound
        if upper_bound is not None:
            non_null_values &= self.df[column] <= upper_bound
        null_values = self.df[column].isna()
        self.df = self.df[non_null_values | null_values]
     
    def drop_column(self, column):
        del self.df[column]
    
    def replace_none_values(self, column, value):
        self.df[column] = self.df[column].fillna(value)
    
    def rename_values(self, column, dict_conversion):
        non_null_values = self.df[column].notna()
        self.df.loc[non_null_values, column] = self.df.loc[non_null_values, column].replace(dict_conversion)
        
    def modify_other_columns(self, column_to_modify, condition_column, operator, target_value, new_value):
        condition = (self.df[condition_column] > target_value) if operator == '>' else (self.df[condition_column] < target_value) if operator == '<' else (self.df[condition_column] == target_value) if operator == '==' else None
        self.df.loc[condition, column_to_modify] = new_value
    
    def remove_incoherent_values(self, column_to_check, reference_column, threshold=0.1):
        non_null_values = self.df[[column_to_check, reference_column]].dropna()
        
        coef = np.polyfit(non_null_values[reference_column], non_null_values[column_to_check], 1)
        expected_values = np.polyval(coef, non_null_values[reference_column])
        
        deviations = abs(non_null_values[column_to_check] - expected_values)
        outlier_values = deviations > threshold * abs(expected_values)
        
        self.df = self.df[~self.df.index.isin(non_null_values[outlier_values].index)]    
        
    def remove_none_values(self, columns):
        self.df = self.df.dropna(subset=columns)   

data = pd.read_json("data/final_dataset.json")
df = pd.DataFrame(data)

dataclean = DataCleaning(df)

print(dataclean.df.shape)

dataclean.drop_column("Fireplace")
dataclean.drop_column("Url")

dataclean.remove_none_values(["PostalCode", "Price", "PropertyId", "TypeOfSale", "TypeOfProperty"])

dataclean.rename_values("PEB", {"A+":"A", "A++":"A", "A_A+": "A"})
dataclean.remove_values("PEB", ["A", "B", "C", "D", "E", "F", "G"])

dataclean.replace_none_values("SwimmingPool", False)
dataclean.replace_none_values("Furnished", False)

dataclean.modify_other_columns("Garden", "GardenArea", ">", 0, True)
dataclean.modify_other_columns("Garden", "GardenArea", ">", 0, True)

dataclean.remove_outliers("ConstructionYear", 1500, 2025)
dataclean.remove_outliers("ShowerCount", 0, 40)
dataclean.remove_outliers("ToiletCount", 0, 40)
dataclean.remove_outliers("ShowerCount", 0, 40)

print(dataclean.df.shape)

dataclean.remove_incoherent_values("SurfaceOfPlot", "Price", 10)
dataclean.remove_incoherent_values("BathroomCount", "RoomCount", 1)
dataclean.remove_incoherent_values("ToiletCount", "RoomCount", 1)
dataclean.remove_incoherent_values("ShowerCount", "RoomCount", 1)
print(dataclean.df["SurfaceOfPlot"].max())

print(dataclean.df.shape)
