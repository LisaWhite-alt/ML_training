"""
Скачайте тренировочный датасэт и  обучите на нём Decision Tree.
После этого скачайте датасэт из задания и предскажите какие наблюдения к кому относятся.
Введите число собачек в вашем датасэте.
"""


# импортируем необходимые библиотеки
from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# загрузим тренировочный и тестовый датасеты
train_data = pd.read_csv("dogs_n_cats.csv")
test_data = pd.read_json("dataset_209691_15.txt")

# проверим, что нет отсутствующих значений
is_null = train_data.isnull().sum()

# в X_train оставляем переменные с важной инфой для обучения модели
X_train = train_data.drop(["Вид"], axis=1)
# в y_train закладываем целевую переменную
y_train = train_data["Вид"]

# сажаем дерево
clf = tree.DecisionTreeClassifier(criterion="entropy")

# обучаем дерево
clf = clf.fit(X_train, y_train)

# предсказываем тестовую целевую переменную
y_test = clf.predict(test_data)

print(list(y_test).count("собачка"))
