from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


titanic_data = pd.read_csv("https://stepic.org/media/attachments/course/524/train.csv")

is_null = titanic_data.isnull().sum() # проверяем на отсутствующие значения

# в Х оставляем переменные с важной инфой для обучения модели
X = titanic_data.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
y = titanic_data.Survived # в y закладываем целевую переменную

X = pd.get_dummies(X) # переводим строковые типы в числовые факторизацией

X = X.fillna({"Age": X.Age.median()}) # заполнение пропущенных значений медианой по Age

clf = tree.DecisionTreeClassifier(criterion="entropy") # сажаем дерево

clf.fit(X, y) # получаем переобученное дерево, фанатично по алгоритму поделившее все явления


# 1. Разделим все данные на тренировочные и тестовые

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf.fit(X_train, y_train) # обучим дерево заново на тренировочных данных
print(clf.score(X_train, y_train)) # посмотрим оценку правильных ответов на исходным данным
print(clf.score(X_test, y_test)) # посмотрим оценку правильных ответов на тестовых данным


# 2. Ограничим глубину дерева

clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)

clf.fit(X_train, y_train) # обучим дерево заново на тренировочных данных
print(clf.score(X_train, y_train)) # посмотрим оценку правильных ответов на исходным данным
print(clf.score(X_test, y_test)) # посмотрим оценку правильных ответов на тестовых данным
