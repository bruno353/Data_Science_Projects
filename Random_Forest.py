#Random Forest model that predict american house prices. melb_data.cvs used.
import pandas as pd
melbourne_data = pd.read_csv("melb_data.csv")
melbourne_data.dropna(axis=0)
Y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
from sklearn.model_selection import train_test_split
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = 0.3, random_state = 1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
model_f = RandomForestRegressor(random_state=1)
model_f.fit(X_treino, Y_treino)
predicao = model_f.predict(X_teste)
print(mean_absolute_error(Y_teste, predicao))
