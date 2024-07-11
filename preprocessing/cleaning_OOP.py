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
        
    def subset_dataframe(self, column, values):
        self.df = self.df[self.df[column].isin(values)]
    
    def merge_df(self, right_df, left_col, right_col):
        self.df = self.df.merge(right_df, how="inner", left_on=left_col, right_on=right_col).drop_duplicates(subset="PropertyId")
        
    def remove_outliers(self, column, lower_bound=None, upper_bound=None, filter_column=None, filter_value=None):
        if filter_column and filter_value:
            filtered_df = self.df[(self.df[filter_column] == filter_value)]
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

data = pd.read_json("data/final_dataset.json")
df = pd.DataFrame(data)
geo_data = pd.read_json('/home/siegfried2021/Bureau/BeCode_AI/Projets/ImmoEliza/Preprocessing-Visualization/preprocessing/georef-belgium-postal-codes.json')
geo_data_df = pd.DataFrame(geo_data)
geo_data_df_short = geo_data_df["postcode"]
geo_data_coord = geo_data_df["geo_point_2d"].apply(pd.Series)
geo_data_df_short = pd.concat([geo_data_df_short.drop(columns=['geo_point_2d']), geo_data_coord], axis=1)

geo_data_csv = geo_data_df.to_csv('data/geo_dataset.csv')
geo_data_short_csv = geo_data_df_short.to_csv('data/geo_short_dataset.csv')

dataclean = DataCleaning(df)

dataclean.drop_column("Fireplace")
dataclean.drop_column("Url")
dataclean.drop_column("District")
dataclean.drop_column("Country")

dataclean.remove_none_values(["PostalCode", "Price", "PropertyId", "TypeOfSale", "TypeOfProperty"])

dataclean.merge_df(geo_data_df_short, "PostalCode", "postcode")
dataclean.drop_column("postcode")

dataclean.subset_dataframe("TypeOfSale", ['residential_sale', 'residential_monthly_rent'])

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

# 2-by-2 correlations for numerical values (with visual matrix)
numerical_df = final_df.select_dtypes(include=['number', 'bool'])
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",annot_kws={"size": 10}, cbar_kws={"shrink": 0.75})

# correlation between construction year and PEB
PEB_encoding = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
PEB_values = ["A", "B", "C", "D", "E", "F", "G"]
final_df["PEB_numerical"] = final_df["PEB"].map(PEB_encoding)
correlation_PEB_Year = final_df[["PEB_numerical", "ConstructionYear"]].corr()
# box plot to visualize the relationship between PEB and construction year
plt.figure(figsize=(10, 6))
sns.boxplot(data=final_df, x="PEB", y="ConstructionYear", order=PEB_values)
plt.xlabel("PEB Score")
plt.ylabel("Construction Year")
plt.title("Relationship between properties construction year and energy performance")

# relationship between house price for sale and presence/absence of a swimming pool
final_df_residential = final_df[final_df['TypeOfSale'] == 'residential_sale']
final_df_house_for_sale = final_df_residential[final_df_residential['TypeOfProperty'] == "House"]
corr_coefficient_pool_price, p_value_pool_price = stats.pointbiserialr(final_df_house_for_sale["SwimmingPool"], final_df_house_for_sale["Price"])
plt.figure(figsize=(10, 6))
sns.boxplot(data=final_df_house_for_sale, x="SwimmingPool", y="Price")
plt.xlabel("Swimming Pool")
plt.ylabel("House Price")

# proportion of swimming pool per house for sale across provinces
proportion_pool_province = final_df_house_for_sale.groupby('Province')['SwimmingPool'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
proportion_pool_province.plot(kind='bar', color='skyblue')
plt.xlabel("Province")
plt.ylabel("Proportion of houses with swimming pool")
plt.title("Proportion of Houses with Swimming Pool per Province")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# house price based on number of facades
plt.figure(figsize=(10, 6))
sns.boxplot(data=final_df, x="NumberOfFacades", y="Price")
plt.xlabel("Number of facades")
plt.ylabel("House Price")

# house and appartment price per province
price_per_province = final_df.groupby(["Province", "TypeOfProperty"])["Price"].mean().astype(float).reset_index()
pivot_table = price_per_province.pivot(index='Province', columns='TypeOfProperty', values='Price')
pivot_table.plot(kind='bar', color=['purple', 'steelblue'])
plt.xlabel("Province")
plt.ylabel("Mean Price of Properties")
plt.title("Mean price of properties per province by property type")
plt.xticks(rotation=45)

# house price based on plot area versus living area
living_area_lower = final_df_house_for_sale['LivingArea'].quantile(0.025)
living_area_upper = final_df_house_for_sale['LivingArea'].quantile(0.975)
surface_of_plot_lower = 50
surface_of_plot_upper = 5000
price_lower_bound = 25000
price_upper_bound = 1250000

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

sns.scatterplot(x='LivingArea', y='Price', data=final_df_house_for_sale, color='purple', ax=axes[0])
sns.regplot(x='LivingArea', y='Price', data=final_df_house_for_sale, scatter=False, color='purple', ax=axes[0])
axes[0].set_title('Living Area vs Price')
axes[0].set_xlabel('Living Area')
axes[0].set_ylabel('Price')
axes[0].set_xlim(living_area_lower, living_area_upper)
axes[0].set_ylim(price_lower_bound, price_upper_bound)

sns.scatterplot(x='SurfaceOfPlot', y='Price', data=final_df_house_for_sale, color='steelblue', ax=axes[1])
sns.regplot(x='SurfaceOfPlot', y='Price', data=final_df_house_for_sale, scatter=False, color='steelblue', ax=axes[1])
axes[1].set_title('Surface of Plot vs Price')
axes[1].set_xlabel('Surface of Plot')
axes[1].set_ylabel('Price')
axes[1].set_xlim(surface_of_plot_lower, surface_of_plot_upper)
axes[1].set_ylim(price_lower_bound, price_upper_bound)
axes[1].set_xscale('log')

fig.suptitle('Comparison of Surface and Living Areas as Predictors on Price with Regression Lines', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])