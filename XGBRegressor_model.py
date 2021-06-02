import pandas as pd
data = pd.read_csv("melb_data.csv")
data.dropna(axis = 0)
X = data[['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt', 'Bathroom', 'Bedroom2', 'Lattitude', 'Longtitude']]
Y = data.Price
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
model.fit(X_train, Y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_test, Y_test)], 
             verbose=False)
predicao = model.predict(X_test
from sklearn.metrics import mean_absolute_error
mean_absolute_error(predicao, Y_test)
