# DataCon_hachaton

Данный репозиторий является решением хакатона(DataCon) от SCAMT ITMO
Задача: Построить предсказательную модель зоны ингибирования молекулы
Порядок выполнения:
1) requirements.py (подгружает пакеты из requirements.txt)
2) united_db.py (Сбор датасета)
3) model_ansamble (Ансамбль из GradienBoostingRegressor, ExtraTreeRegressor, RandomForestRegressor),
   moodel_stack (Стеккинг из GradienBoostingRegressor, RandomForestRegressor
   model_extra_trees (ExtraTreeRegressor)

4) Веса моделей: model_ansamble.pkl, moodel_stack.pkl   
    
Над задачей работали: Карташов Игорь, Валеева Лилиана, Шестун Павел
