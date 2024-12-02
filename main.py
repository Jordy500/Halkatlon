import pandas as pd


df = pd.read_csv('database.csv')

ndf = df.drop(['ID','Year_Birth','Z_Revenue', 'Z_CostContact', 'Complain','Education', 'Marital_Status', 'Dt_Customer'], axis=1, inplace=True)

print(df.info())
print(df.describe())
print(df.head())

