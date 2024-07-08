#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
#%%
# get and prep data
data_root = "https://github.com/ageron/data/raw/main/"
life_sat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x = life_sat[["GDP per capita (USD)"]].values
y = life_sat[["Life satisfaction"]].values
print(x)
print(y)
#%%
# visualize data
life_sat.plot(kind='scatter', grid=True,
              x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23500, 62500, 4, 9])
plt.show()
#%%
# select a linear model
model_LR = LinearRegression()
model_KNR = KNeighborsRegressor(n_neighbors=3)

# train the model
model_LR.fit(x, y)
model_KNR.fit(x, y)

# make a prediction for cyprus
X_new = [[37_655.2]]    # cyprus GDP per capita in 2020
print("Linear Regression ", model_LR.predict(X_new))
print("KNN Regression ", model_KNR.predict(X_new))
#%%
