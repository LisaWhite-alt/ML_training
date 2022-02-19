import pandas as pd
from sklearn.tree import DecisionTreeClassifier


train_data_tree = pd.read_csv("train_data_tree.csv")

clf = DecisionTreeClassifier(criterion="entropy")

x = train_data_tree[["sex", "exang"]]
y = train_data_tree.num

clf.fit(x, y)
e = clf.tree_.impurity[0] # энтропия в корне дерева
l_node = clf.tree_.children_left[0] # индекс корня левого поддерева
n1 = clf.tree_.n_node_samples[l_node] # сэмплов в левом поддереве
e1 = clf.tree_.impurity[l_node] # энтропия в корне левого поддерева
r_node = clf.tree_.children_right[0] # индекс корня правого поддерева
n2 = clf.tree_.n_node_samples[r_node] # сэмплов в правом поддереве
e2 = clf.tree_.impurity[r_node] # энтропия в корне правого поддерева

ig = e - (n1*e1 + n2*e2)/(n1+n2)

print(ig)