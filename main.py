import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv('database.csv')

ndf = df.drop(['ID','Year_Birth','Z_Revenue', 'Z_CostContact', 'Complain','Dt_Customer'], axis=1, inplace=True)
dt_clear = df.dropna(subset=['Income'])

x = dt_clear.drop(columns=['Response'])
y = dt_clear['Response']

x = pd.get_dummies(x, columns=['Education', 'Marital_Status'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))



print(df.info())
print(df.describe())
print(df.head())
print(dt_clear.info())
print(dt_clear.head())
print("Précision :", accuracy_score(y_test, y_pred))


