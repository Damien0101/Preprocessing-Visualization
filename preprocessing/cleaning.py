import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_json("data/final_dataset.json")
df = pd.DataFrame(data)
# data_csv = df.to_csv('data/dataset.csv')

# 1. Column URL
df = df.drop(columns=['Url'])

# 2. Column PostCode : merging with dataframe of belgian postcodes
file_postcodes = "preprocessing/georef-belgium-postal-codes.json"
df_postcodes = pd.read_json(file_postcodes)
df_postcodes = df_postcodes[["postcode"]]
df = df.merge(df_postcodes, how="inner", right_on="postcode", left_on="PostalCode").drop_duplicates()
del df["postcode"]

# 3. Column Swimming Pool : filling null values for SwimmingPool by False, since no mention of swimming pool is absence of swimming pool
df["SwimmingPool"] = df["SwimmingPool"].fillna(False)

# 4. Column Fireplace : suggestion to delete the fireplace column, because of the absence of True values
del df["Fireplace"]

# 5. Column Type of Sale : cleaning to get rid of unwanted sales types
unwanted_sales = ["annuity_monthly_amount", "annuity_without_lump_sum", "homes_to_build", "annuity_lump_sum"]
df = df[~df["TypeOfSale"].isin(unwanted_sales)]

# cleaning PEB values to have values from A to G
df["PEB"] = df["PEB"].replace({"A+":"A", "A++":"A", "A_A+": "A"})
PEB_values = ["A", "B", "C", "D", "E", "F", "G"]
df["PEB"] = df["PEB"].apply(lambda x: x if x in PEB_values else None)

# cleaning construction year values
df["ConstructionYear"] = df["ConstructionYear"].apply(lambda x: x if x <= 2025 else np.nan)

# correlation between construction year and PEB
PEB_encoding = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df["PEB_numerical"] = df["PEB"].map(PEB_encoding)
correlation_PEB_Year = df[["PEB_numerical", "ConstructionYear"]].corr()

# box plot to visualize the relationship between PEB and construction year
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="PEB", y="ConstructionYear", order=PEB_values)
plt.xlabel("PEB Score")
plt.ylabel("Construction Year")
plt.title("Relationship between properties construction year and energy performance")

# work around the swimming pool

# relationship between house price for sale and presence/absence of a swimming pool
df_residential = df[df['TypeOfSale'] == 'residential_sale']
df_house_for_sale = df_residential[df_residential['TypeOfProperty'] == 1]
corr_coefficient_pool_price, p_value_pool_price = stats.pointbiserialr(df_house_for_sale["SwimmingPool"], df_house_for_sale["Price"])

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_house_for_sale, x="SwimmingPool", y="Price")
plt.xlabel("Swimming Pool")
plt.ylabel("House Price")

# proportion of swimming pool per house for sale across provinces
proportion_pool_province = df_house_for_sale.groupby('Province')['SwimmingPool'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
proportion_pool_province.plot(kind='bar', color='skyblue')
plt.xlabel("Province")
plt.ylabel("Proportion of houses with swimming pool")
plt.title("Proportion of Houses with Swimming Pool per Province")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

print(df[df["SurfaceOfPlot"] == 950774])