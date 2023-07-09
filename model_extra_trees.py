import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing
import shap

# Загружаем датасет
data = pd.read_csv('https://raw.githubusercontent.com/IGK-arch/DataCon_hachaton/main/final.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)

# Нормализуем датасет с помощью MinMaxScaller
to_scale_df = data.copy()
scaled_data = preprocessing.MinMaxScaler().fit_transform(to_scale_df)
scaled_df = pd.DataFrame(scaled_data, columns=to_scale_df.columns)

# разделяем таргет и фичи
X_norm = scaled_df.drop('ZOI_drug_NP', axis=1)
y_norm = scaled_df['ZOI_drug_NP']

# Разделяем выборки на train и test
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm,
                                                    train_size=0.8,
                                                    random_state=777)

# Перемешаем датасет более качественного обучения
shuffled_df = scaled_df.sample(frac=1, random_state=777)
shuffled_df_X = shuffled_df.drop('ZOI_drug_NP', axis=1)
shuffled_df_y = shuffled_df['ZOI_drug_NP']

# обучаем ExtraTreesRegressor
model_xtra = ExtraTreesRegressor(n_estimators=170)

param_grid = {'n_estimators': range(100, 400, 50),
              'max_features': range(50, 401, 50),
              'min_samples_leaf': range(20, 50, 10),
              'min_samples_split': range(15, 36, 5),
              }

xtra_grid = GridSearchCV(model_xtra, param_grid, cv=5)
xtra_grid.fit(shuffled_df_X, shuffled_df_y)
best_params = xtra_grid.best_params_
print(f'best grid search params: {best_params}')
# n_estimators=300, 
model_xtra = ExtraTreesRegressor(**best_params)
score_extra_tree = cross_val_score(model_xtra, shuffled_df_X, shuffled_df_y, cv=5)

print('Cross validation scores: ', score_extra_tree)
print('Mean Score: ', score_extra_tree.mean())

# Отнормализуем обратно X_test для прогноза
min_val = data.ZOI_drug_NP.min()
max_val = data.ZOI_drug_NP.max()
y_true = y_test * (max_val - min_val) + min_val
y_pred = model_xtra.predict(X_test) * (max_val - min_val) + min_val

# Проверим r2 score для тестовых данных
print('r2 score: ', r2_score(y_true, y_pred))
# Проверим mse score для тестовых данных
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print('mse: ', mse)
print('rmse: ', rmse)

# Построим график сравнения y_test y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, color='red', label='Test Data')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.title('ExtraTreesRegressor: Real vs Predicted (Test Data)')
plt.legend()
plt.show()

# Построим график сравнения y_test y_pred
# Может потребоваться какое то время для выполнения
def feature_importance():
    explainer = shap.TreeExplainer(model_xtra)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, plot_type='bar')