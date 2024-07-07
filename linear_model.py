import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# get and prep data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# visualize data
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# select a linear model
model_LR = LinearRegression()
model_KNR = KNeighborsRegressor(n_neighbors=3)

# train the model
model_LR.fit(x, y)
model_KNR.fit(x, y)

# make a prediction for cyprus
X_new = [[37_655.2]] #cyprus GDP per capita in 2020
print("Linear Regression ", model_LR.predict(X_new))
print("KNN Regression ", model_KNR.predict(X_new))
