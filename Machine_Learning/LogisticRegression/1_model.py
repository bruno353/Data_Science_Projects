import pandas as pd
df = pd.read_csv('HR_comma_sep.csv')
Y = df.left
dummies = pd.get_dummies(df.salary)
merged = pd.concat([df, dummies], axis = "columns")
#precisa agora dropar a coluna suburb e uma coluna dummie criada, pode ser qualquer uma. (para saber mais pesquise dummie variable trap).
df = merged.drop(['salary', 'low'], axis = 'columns')
X = df.drop(["left", "Department", ], axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2 , random_state = 2 )
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
