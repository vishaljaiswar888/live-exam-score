import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def marks_prediction(marks):
    x = pd.read_csv("static/Linear_X_Train.csv")
    y = pd.read_csv("static/Linear_Y_Train.csv")

    X = x.values
    y = y.values

    model = LinearRegression()
    model.fit(X, y)

    X_test = np.array(marks)
    X_test = X_test.reshape((1, -1))

    return model.predict(X_test)[0]

