
import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

with mlflow.start_run():
    data = 'vendas_sorvete_dataset.csv'
    sorvetes = pd.read_csv(data)

    x, y = sorvetes[['temperatura']].values, sorvetes['vendas'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    model = LinearRegression().fit(x_train, y_train)

    y_hat = model.predict(x_test)

    mse = mean_squared_error(y_test, y_hat)
    rmse = np.sqrt(mse)                         #taking the square root to improve understanding
    r2 = r2_score(y_test, y_hat)

    print(f'RMSE: {rmse:.2f}')
    print(f'R^2: {r2:.4f}')

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2_score", r2)

    mlflow.sklearn.log_model(model, "ice_cream_model")

