



import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_density, geom_line, geom_point, ggtitle
import math

# Modeling preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline

# Modeling and resampling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

#%%
bank = pd.read_csv("data/bank-additional-full.csv", sep= ";")


#%%
bank.sample(10)
bank.shape

# %%

y = bank["y"]
train, test = train_test_split(bank, test_size=0.3, 
random_state=123, stratify=y)

# %%

f"raw data dimensions: {bank.shape}; training dimensions: {train.shape}; testing dimensions:  {test.shape}"

# %%

# Extract features and response
features = train.drop(columns="y")
label = train["y"]



#%%
results = grid_search.fit(features, label)
#%%

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, selector(dtype_include="object"))])

knn_fit = Pipeline(steps=[('preprocessor', preprocessor),
                          ('knn', KNeighborsRegressor(metric='euclidean'))])
# %%
# Specify resampling strategy
cv = RepeatedKFold(n_splits=10, n_repeats=5)
# %%
# Create grid of hyperparameter values
hyper_grid = {'knn__n_neighbors': range(3, 26)}
#%%

# Tune a knn model using grid search
grid_search = GridSearchCV(knn_fit, hyper_grid, cv=cv, scoring='neg_mean_squared_error')
#%%


# %%
