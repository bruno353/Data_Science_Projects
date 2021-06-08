  import pandas as pd
	dados = pd.read_csv("advertising.csv") //quero criar uma relação entre os gastos com publicidade em tv, radio e jornal e quanto de vendas teve por conta desse investimento.
	#features:
	x = [["TV","Radio","Newspaper"]]
	y = ["Sales"]

	#train and test:
	from sklearn.model_selection import train_test_split
	x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3, random_state = 2)


	#importando biblioteca sklearn:
	from sklearn.linear_model import LinearRegression

	#instanciando o objeto do modelo:
	modelo_linear = LinearRegression()
	
	#fit no modelo:
	modelo_linear.fit(x_treino, y_treino)
	
	#intercepto e coeficiente: // y = ax + b ( intercepto é o b, coeficiente é o a)
	print(modelo_linear.intercept_) 
	print(modelo_linear.coef_)
	
	#predizendo os valores:
	predicoes = modelo_linear.predict(x_teste)

	#importando as bibliotecas de avaliação do modelo
	from sklearn.metrics import mean_absolute_error
	from sklearn.metrics import mean_squared_error	
	from sklearn.metrics import r2_score
	import numpy as np
	#vai analisar entre as predicoes o teste para ver a acurácia
	print(mean_absolute_error(y_teste, predicoes)) 
	print(mean_squared_error(y_teste, predicoes))
	print(np.sqrt(mean_squared_error(y_teste, predicoes)) //jogando ao quadrado a função
	print(r2_score(y_teste, predicoes))
