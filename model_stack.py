import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing
import shap
import pickle

# Загружаем датасет
data = pd.read_csv('https://raw.githubusercontent.com/IGK-arch/DataCon_hachaton/main/final.csv')
data.drop(['Unnamed: 0', ], axis=1, inplace=True)

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

# обучаем GradientBoostingRegressor
model_grboost = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.036, subsample=0.6, alpha=0.25)
scores2 = cross_val_score(model_grboost, shuffled_df_X, shuffled_df_y, cv=5,)
print('Cross validation scores: ', scores2)
print('Mean Score: ', scores2.mean())
model_grboost.fit(X_train, y_train)

# обучаем RandomForestRegressor
model_forest = RandomForestRegressor(n_estimators=150,)
scores3 = cross_val_score(model_forest, shuffled_df_X, shuffled_df_y, cv=5,)
print(f'mean score RandomForestRegressor CV: {scores3.mean()}')
model_forest.fit(X_train, y_train)

# cтекинг GradientBoostingRegressor и RandomForestRegressor
from sklearn.ensemble import StackingRegressor
estims2 = [('gbr', model_grboost), ('forest', model_forest)]
stack = StackingRegressor(estimators=estims2, cv=5)
stack.fit(X_train, y_train)
score_stack = cross_val_score(stack, shuffled_df_X, shuffled_df_y, cv=5)
print('Cross validation scores: ', score_stack)
print('Mean Score: ', score_stack.mean())

# Отнормализуем обратно X_test для прогноза
min_val = data.ZOI_drug_NP.min()
max_val = data.ZOI_drug_NP.max()
y_true = y_test * (max_val - min_val) + min_val
y_pred = stack.predict(X_test) * (max_val - min_val) + min_val

# Проверим r2 score для тестовых данных
print('r2 score: ', r2_score(y_true, y_pred))
# Проверим mse score для тестовых данных
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print('mse: ', mse)
print('rmse: ', rmse )

# Построим график сравнения y_test y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, color='red', label='Test Data')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.title('StackingRegressor: Real vs Predicted (Test Data)')
plt.legend()
plt.show()


# Построим график сравнения y_test y_pred
# Может потребоваться какое то время для выполнения
def feature_importance():
    explainer = shap.Explainer(stack.predict, X_train, max_evals=1041)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, plot_type='bar')


# Save weights
def weights_model():
    with open("model_stack.pkl", "wb") as f:
        pickle.dump(stack, f)


weights_model()