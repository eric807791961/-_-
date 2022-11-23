# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:42:47 2021

@author: erich
"""
import pandas as pd
import numpy as np
from catboost import Pool
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"D:\Python data\new_df.csv", index_col=0)
df.describe()
df['parking area'] = np.log10(df['parking area'] + 1)
df['House Age'] = np.log10(df['House Age'] + 1)

df1 = df[df['住家用']==1]

train_set, validation_set, test_set = np.split(df1.sample(frac=1), [int(.6*len(df1)), int(.8*len(df1))])

X_train = train_set.drop(['unit_price'],axis = 1)
Y_train = train_set['unit_price']
X_validation = validation_set.drop(['unit_price'],axis = 1)
Y_validation = validation_set['unit_price']
X_test = test_set.drop(['unit_price'],axis = 1)
Y_test = test_set['unit_price']


"""
Catboost
"""
Cat_columns = [3,4,5,6,7,8,9,11,15,55,58]
Cat_train = X_train.iloc[:, Cat_columns]
Cat_validation = X_validation.iloc[:, Cat_columns]
Cat_test = X_test.iloc[:, Cat_columns]
catboost_model = CatBoostRegressor(
    iterations=100,
    max_ctr_complexity=15,
    random_seed=42,
    od_type='Iter',
    od_wait=25,
    verbose=50,
    depth=16
)

cat_features = list(range(1,11))

catboost_model.fit(
    Cat_train, Y_train,
    cat_features=None,
    eval_set=(Cat_validation, Y_validation)
)

plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
#將預設字形設定為從網路下載好放入ttf資料夾的ttf檔案

feature_score = pd.DataFrame(list(zip(Cat_train.dtypes.index, catboost_model.get_feature_importance(Pool(Cat_train, label=Y_train, cat_features=None)))), columns=['Feature','Score'])
feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')

plt.rcParams["figure.figsize"] = (19, 6)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
ax.set_xlabel('')
rects = ax.patches
labels = feature_score['Score'].round(2)

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')

plt.show()

catboost_train_pred = catboost_model.predict(Cat_train)
catboost_val_pred = catboost_model.predict(Cat_validation)
catboost_test_pred = catboost_model.predict(Cat_test)

def model_performance_sc_plot(predictions, labels, title):
    # Get min and max values of the predictions and labels.
    min_val = max(max(predictions), max(labels))
    max_val = min(min(predictions), min(labels))
    # Create dataframe with predicitons and labels.
    performance_df = pd.DataFrame({"Label":labels})
    performance_df["Prediction"] = predictions
    # Plot data
    sns.jointplot(y="Label", x="Prediction", data=performance_df, kind="reg", height=7)
    plt.plot([min_val, max_val], [min_val, max_val], 'm--')
    plt.title(title, fontsize=9)
    plt.show()
    
# model_performance_sc_plot(catboost_train_pred, Y_train, 'Train')
model_performance_sc_plot(catboost_val_pred, Y_validation, 'Validation')

print('Train rmse:', np.sqrt(mean_squared_error(Y_train, catboost_train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(Y_validation, catboost_val_pred)))



"""
XGBoost
"""

xgb_model = XGBRegressor(max_depth=15, 
                         n_estimators=750, 
                         min_child_weight=1000,  
                         colsample_bytree=0.7, 
                         subsample=0.7, 
                         eta=0.3, 
                         seed=0)

xgb_model.fit(X_train, 
              Y_train, 
              eval_metric="rmse", 
              eval_set=[(X_train, Y_train), (X_validation, Y_validation)], 
              verbose=20, 
              early_stopping_rounds=20)

plt.rcParams["figure.figsize"] = (15, 6)
plot_importance(xgb_model)
plt.show()

xgb_train_pred = xgb_model.predict(X_train)
xgb_val_pred = xgb_model.predict(X_validation)
xgb_test_pred = xgb_model.predict(X_test)

model_performance_sc_plot(xgb_val_pred, Y_validation, 'Validation')

"""
Random Forest
"""

rf_model = RandomForestRegressor(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1)
rf_model.fit(X_train, Y_train)

rf_train_pred = rf_model.predict(X_train)
rf_val_pred = rf_model.predict(X_validation)
rf_test_pred = rf_model.predict(X_test)

model_performance_sc_plot(rf_val_pred, Y_validation, 'Validation')

"""
Linear Regression
"""

lr_model = LinearRegression(n_jobs=-1)
lr_model.fit(X_train, Y_train)

lr_train_pred = lr_model.predict(X_train)
lr_val_pred = lr_model.predict(X_validation)
lr_test_pred = lr_model.predict(X_test)

"""
KNN Regressor
"""

X_train_sampled = X_train[:200000]
Y_train_sampled = Y_train[:100000]

knn_model = KNeighborsRegressor(n_neighbors=9, leaf_size=13, n_jobs=-1)
knn_model.fit(X_train, Y_train)

knn_train_pred = knn_model.predict(X_train)
knn_val_pred = knn_model.predict(X_validation)
knn_test_pred = knn_model.predict(X_test)

model_performance_sc_plot(knn_val_pred, Y_validation, 'Validation')

"""
create first layer
"""

first_level = pd.DataFrame(catboost_val_pred, columns=['catboost'])
first_level['xgbm'] = xgb_val_pred
first_level['random_forest'] = rf_val_pred
first_level['linear_regression'] = lr_val_pred
first_level['knn'] = knn_val_pred
first_level['label'] = Y_validation.values

first_level_test = pd.DataFrame(catboost_test_pred, columns=['catboost'])
first_level_test['xgbm'] = xgb_test_pred
first_level_test['random_forest'] = rf_test_pred
first_level_test['linear_regression'] = lr_test_pred
first_level_test['knn'] = knn_test_pred
first_level_test.head()

"""
Ensemble
"""

meta_model = LinearRegression(n_jobs=-1)
first_level.drop('label', axis=1, inplace=True)
meta_model.fit(first_level, Y_validation)

ensemble_pred = meta_model.predict(first_level)
final_predictions = meta_model.predict(first_level_test)

model_performance_sc_plot(ensemble_pred, Y_validation, 'Validation')

print('rmse:', np.sqrt(mean_squared_error(ensemble_pred, Y_validation)))
