![](intro.gif)

# DataCon_hachaton

Данный репозиторий является решением [хакатона DataCon от SCAMT ITMO](https://github.com/dataconHack/hackathon)

Задача: Построить предсказательную модель зоны ингибирования молекулы

Порядок выполнения:
1) requirements.py (подгружает пакеты из requirements.txt)
2) united_db.py (Сбор датасета)
3) model_ansamble (Ансамбль из GradienBoostingRegressor, ExtraTreeRegressor, RandomForestRegressor),
   
   moodel_stack (Стеккинг из GradienBoostingRegressor, RandomForestRegressor
   
   model_extra_trees (ExtraTreeRegressor)

5) Веса моделей: model_ansamble.pkl, moodel_stack.pkl
6) Датасеты:
   
             -db1(data)
   
             -db2(bacterial_descriptors)
   
             -db3(drug_descriptors)
   
             -general_data (united db1,db2,db3)
   
             -final (итоговый очищенный датасет)
   
Над задачей работали: 

                     Карташов Игорь

                     Валеева Лилиана @seym0
                     
                     Шестун Павел
                     
