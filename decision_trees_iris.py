"""
Скачайте датасэт с ирисами, обучите деревья с глубиной от 1 до 100.
Целевой переменной при обучении является переменная species.
Затем визуализируйте зависимость скора и предсказания от глубины дерева.
"""

# импортируем необходимые библиотеки
from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# загрузим тренировочный и тестовый датасеты
train_iris_data = pd.read_csv("train_iris.csv")
test_iris_data = pd.read_csv("test_iris.csv")

np.random.seed(0)

# определим независимые и целевые переменные
X_train = train_iris_data[["sepal length", "sepal width", "petal length", "petal width"]]
X_test = test_iris_data[["sepal length", "sepal width", "petal length", "petal width"]]
y_train = train_iris_data.species
y_test = test_iris_data.species

max_deth_values = range(1, 100)
scores_data = pd.DataFrame()

for max_depth in max_deth_values:
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    temp_score_data = pd.DataFrame({
        "max_depth": [max_depth],
        "train_score": [train_score],
        "test_score": [test_score]
    })
    scores_data = pd.concat([scores_data, temp_score_data], axis=0)

scores_data_long = pd.melt(
    scores_data,
    id_vars=["max_depth"],
    value_vars=["train_score", "test_score"],
    var_name="set_type",
    value_name="score")

plot = sns.lineplot(x="max_depth", y="score", hue="set_type", data=scores_data_long)

print(plot)
