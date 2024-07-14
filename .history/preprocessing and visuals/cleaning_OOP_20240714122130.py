import pandas as pd 
import numpy as np
class DataCleaning:    
    def __init__(self, df):
        self.df = df
    
    def remove_values(self, column, values):
        self.df[column] = self.df[column].apply(lambda x: x if x in values else None)
        
    def subset_dataframe(self, column, values):
        self.df = self.df[self.df[column].isin(values)]
    
    def merge_df(self, right_df, left_col, right_col):
        self.df = self.df.merge(right_df, how="inner", left_on=left_col, right_on=right_col).drop_duplicates(subset="PropertyId")
        
    def remove_outliers(self, column, lower_bound=None, upper_bound=None, filter_column=None, filter_value=None):
        if filter_column and filter_value:
            filtered_df = self.df[self.df[filter_column] == filter_value]
            non_null_values = filtered_df[column].notna()
            if lower_bound is not None:
                non_null_values &= (filtered_df[column] >= lower_bound)
            if upper_bound is not None:
                non_null_values &= (filtered_df[column] <= upper_bound)
            
            outliers = filtered_df[~non_null_values]
            self.df = self.df.drop(outliers.index)
        
        else:
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
        self.df[column] = self.df[column].replace(dict_conversion)
        
    def modify_other_columns(self, column_to_modify, condition_column, operator, target_value, new_value):
        condition = (self.df[condition_column] > target_value) if operator == '>' else \
                    (self.df[condition_column] < target_value) if operator == '<' else \
                    (self.df[condition_column] == target_value) if operator == '==' else None
        self.df.loc[condition, column_to_modify] = new_value
    
    def remove_incoherent_values(self, column_to_check, reference_column, threshold, filter_column=None, filter_value=None):
        if filter_column and filter_value:
            filtered_df = self.df[self.df[filter_column] == filter_value]
        else:
            filtered_df = self.df
            
        non_null_values = filtered_df[[column_to_check, reference_column]].dropna()
        
        coef = np.polyfit(non_null_values[reference_column], non_null_values[column_to_check], 1)
        expected_values = np.polyval(coef, non_null_values[reference_column])
        
        deviations = abs(non_null_values[column_to_check] - expected_values)
        outlier_values = deviations > threshold * abs(expected_values)
        
        self.df = self.df[~self.df.index.isin(filtered_df.index[filtered_df.index.isin(non_null_values[outlier_values].index)])]
        
    def remove_none_values(self, columns):
        self.df[columns] = self.df[columns].dropna()
    
    def convert_to_numbers(self, column, dict_conversion):
        self.df[f"{column}_numerical"] = self.df[column].map(dict_conversion)

data = pd.read_json("/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/Preprocessing-Visualization/data/final_dataset.json")
df = pd.DataFrame(data)
geo_data = pd.read_json('/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/Preprocessing-Visualization/data/georef-belgium-postal-codes.json')
geo_data_df = pd.DataFrame(geo_data)
geo_data_df = geo_data_df["postcode"]
geo_data_csv = geo_data_df.to_csv('data/geo_dataset.csv')

dataclean = DataCleaning(df)

print(dataclean.df.shape)

dataclean.drop_column("Fireplace")
dataclean.drop_column("Url")
dataclean.drop_column("Country")

dataclean.remove_none_values(["PostalCode", "Price", "PropertyId", "TypeOfSale", "TypeOfProperty"])

print(dataclean.df.shape)

dataclean.merge_df(geo_data_df, "PostalCode", "postcode")
dataclean.drop_column("postcode")

print(dataclean.df.shape)

dataclean.subset_dataframe("TypeOfSale", ['residential_sale', 'residential_monthly_rent'])

print(dataclean.df.shape)

dataclean.rename_values("PEB", {"A+":"A", "A++":"A", "A_A+": "A"})
dataclean.rename_values("NumberOfFacades", {1:2})
dataclean.rename_values("TypeOfProperty", {1:"House", 2:"Apartment"})

dataclean.remove_values("PEB", ["A", "B", "C", "D", "E", "F", "G"])

dataclean.replace_none_values("SwimmingPool", 0)
dataclean.replace_none_values("Furnished", 0)
dataclean.replace_none_values("Garden", 0)
dataclean.replace_none_values("Terrace", 0)

dataclean.modify_other_columns("Garden", "GardenArea", ">", 0, 1)

dataclean.remove_outliers("Price", 10000, 25000000, "TypeOfSale", "residential_sale")
dataclean.remove_outliers("Price", 1000, 50000, "TypeOfSale", "residential_monthly_rent")
dataclean.remove_outliers("ConstructionYear", 1500, 2025)
dataclean.remove_outliers("LivingArea", 5, 5000)
dataclean.remove_outliers("ShowerCount", 0, 40)
dataclean.remove_outliers("ToiletCount", 0, 40)
dataclean.remove_outliers("ShowerCount", 0, 40)
dataclean.remove_outliers("NumberOfFacades", 1, 4)

dataclean.remove_incoherent_values("SurfaceOfPlot", "Price", 1, "TypeOfSale", "residential_sale")
dataclean.remove_incoherent_values("SurfaceOfPlot", "Price", 1, "TypeOfSale", "residential_monthly_rent")
dataclean.remove_incoherent_values("LivingArea", "Price", 1, "TypeOfSale", "residential_sale")
dataclean.remove_incoherent_values("LivingArea", "Price", 1, "TypeOfSale", "residential_monthly_rent")
dataclean.remove_incoherent_values("RoomCount", "Price", 1)
dataclean.remove_incoherent_values("LivingArea", "RoomCount", 1)
dataclean.remove_incoherent_values("BathroomCount", "RoomCount", 1)
dataclean.remove_incoherent_values("ToiletCount", "RoomCount", 1)
dataclean.remove_incoherent_values("ShowerCount", "RoomCount", 1)

dataclean.convert_to_numbers("PEB", {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
dataclean.convert_to_numbers("Kitchen", {'NOT_INSTALLED': 1, 'USA_UNINSTALLED': 2, 'INSTALLED': 3, 'USA_INSTALLED': 4, 'SEMI_EQUIPPED': 5, 'USA_SEMI_INSTALLED': 6, 'HYPER_EQUIPPED': 7, 'USA_HYPER_EQUIPPED': 8})
dataclean.convert_to_numbers("StateOfBuilding", {'TO_BE_DONE_UP': 1, 'TO_RESTORE': 2, 'TO_RENOVATE': 3, 'GOOD': 4, 'JUST_RENOVATED': 5, 'AS_NEW': 6})
dataclean.convert_to_numbers("TypeOfProperty", {"House": 1, "Apartment": 2})
dataclean.convert_to_numbers("TypeOfSale", {"residential_sale": 1, "residential_monthly_rent": 2})
dataclean.convert_to_numbers("Region", {"Flanders": 1, "Wallonie": 2, "Brussels": 3})
dataclean.convert_to_numbers("Province", {"Walloon Brabant": 1, "Hainaut": 2, "Namur": 3, "Liège": 4, "Luxembourg": 5, "Brussels": 6, "Flemish Brabant": 7, "West Flanders": 8, "East Flanders": 9, "Antwerp": 10, "Limburg": 11})

final_df = dataclean.df
final_csv = final_df.to_csv('data/final_set.csv')