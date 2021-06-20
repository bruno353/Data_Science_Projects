#analyze and predict house prices from melb_data.csv
import pandas as pd
df = pd.read_csv('melb_data.csv')
df.head()
#data cleaning
df.isna().sum()
df.Car.fillna(0, inplace = True)
df.drop(['CouncilArea'], axis = 1, inplace = True)
df.drop(['BuildingArea'], axis = 1, inplace = True)
df.head()
df.drop(['YearBuilt'], axis = 1, inplace = True)
df.drop(['Address'], axis = 1, inplace = True)
df = pd.get_dummies(df, drop_first = True)
X = df.drop(['Price'], axis = 1)
Y = df.Price
#data modeling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)
model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error
predict = model.predict(X_test)
#results according to absolute error
print(mean_absolute_error(y_test, predict))
