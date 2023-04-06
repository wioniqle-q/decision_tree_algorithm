import csv
import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  
        self.threshold = threshold         
        self.left = left                    
        self.right = right                 
        self.value = value                 

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, data):
        features = list(range(len(data[0]) - 1))
        self.tree = self._build_tree(data, features, 0)

    def _build_tree(self, data, features, depth):
        labels = [sample[-1] for sample in data]
        if depth == self.max_depth or len(set(labels)) == 1:
            return Node(value=max(set(labels), key=labels.count))

        best_feature_index, best_threshold = self._find_best_split(data, features)
        left_data, right_data = self._split_data(data, best_feature_index, best_threshold)

        left_tree = self._build_tree(left_data, features, depth + 1)
        right_tree = self._build_tree(right_data, features, depth + 1)

        return Node(feature_index=best_feature_index, threshold=best_threshold, left=left_tree, right=right_tree)

    def _find_best_split(self, data, features):
        best_gain = 0
        best_feature_index = None
        best_threshold = None

        for feature_index in features:
            values = [sample[feature_index] for sample in data]
            thresholds = list(set(values))

            for threshold in thresholds:
                left_data, right_data = self._split_data(data, feature_index, threshold)
                if not left_data or not right_data:
                    continue

                gain = self._information_gain(left_data, right_data)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _split_data(self, data, feature_index, threshold):
        left_data = []
        right_data = []

        for sample in data:
            if sample[feature_index] < threshold:
                left_data.append(sample)
            else:
                right_data.append(sample)

        return left_data, right_data

    def _information_gain(self, left_data, right_data):
        parent_entropy = self._entropy(left_data + right_data)
        left_entropy = self._entropy(left_data)
        right_entropy = self._entropy(right_data)

        num_left = len(left_data)
        num_right = len(right_data)
        total = num_left + num_right

        gain = parent_entropy - (num_left / total) * left_entropy - (num_right / total) * right_entropy

        return gain

    def _entropy(self, data):
        labels = [sample[-1] for sample in data]
        _, counts = np.unique(labels, return_counts=True)

        probabilities = counts / len(labels)
        entropy = sum(probabilities * -np.log2(probabilities))

        return entropy

    def predict(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

    def __repr__(self):
        return str(self.tree)

test_data = []
with open('testSet.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) 
    for row in reader:
        test_data.append([float(val) if val.replace('.', '').isdigit() else val for val in row])

train_data = []
with open('trainSet.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) 
    for row in reader:
        train_data.append([float(val) if val.replace('.', '').isdigit() else val for val in row])

dt = DecisionTree()
dt.fit(train_data)

tp_test, tn_test, fp_test, fn_test = 0, 0, 0, 0
for sample in test_data:
    prediction = dt.predict(sample[:-1])
    if prediction == sample[-1]:
        if prediction == 'good':
            tp_test += 1
        else:
            tn_test += 1
    else:
        if prediction == 'good':
            fp_test += 1
        else:
            fn_test += 1

test_accuracy = (tp_test + tn_test) / (tp_test + tn_test + fp_test + fn_test)
test_tp_rate = tp_test / (tp_test + fn_test)
test_tn_rate = tn_test / (tn_test + fp_test)
test_tp_count = tp_test + fn_test
test_tn_count = tn_test + fp_test

tp_train, tn_train, fp_train, fn_train = 0, 0, 0, 0
for sample in train_data:
    prediction = dt.predict(sample[:-1])
    if prediction == sample[-1]:
        if prediction == 'good':
            tp_train += 1
        else:
            tn_train += 1
    else:
        if prediction == 'good':
            fp_train += 1
        else:
            fn_train += 1
        
train_accuracy = (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
train_tp_rate = tp_train / (tp_train + fn_train)
train_tn_rate = tn_train / (tn_train + fp_train)
train_tp_count = tp_train + fn_train
train_tn_count = tn_train + fp_train

with open('performance_metrics_en.txt', 'a', encoding="utf-8") as f:
    f.write('Train Performance Metrics\n')
    f.write('Accuracy: {:.3f}\n'.format(train_accuracy))
    f.write('TP rate: {:.3f}\n'.format(train_tp_rate))
    f.write('TN rate: {:.3f}\n'.format(train_tn_rate))
    f.write('TP count: {}\n'.format(train_tp_count))
    f.write('TN count: {}\n'.format(train_tn_count))
    f.write('\n')   

    f.write('Test Performance Metrics\n')
    f.write('Accuracy: {:.3f}\n'.format(test_accuracy))
    f.write('TP rate: {:.3f}\n'.format(test_tp_rate))
    f.write('TN rate: {:.3f}\n'.format(test_tn_rate))
    f.write('TP count: {}\n'.format(test_tp_count))
    f.write('TN count: {}\n'.format(test_tn_count))

with open('performance_metrics_tr.txt', 'w', encoding="utf-8") as f:
     f.write('Eğitim (Train) sonucu:\n')
     f.write('Accuracy: {:.3f}\n'.format(train_accuracy))
     f.write('TPrate: {:.3f}\n'.format(train_tp_rate))
     f.write('TNrate: {:.3f}\n'.format(train_tn_rate))
     f.write('TP adedi: {}\n'.format(train_tp_count))
     f.write('TN adedi: {}\n'.format(train_tn_count))
     f.write('\n')

     f.write('Sınama (Test) sonucu:\n')
     f.write('Accuracy: {:.3f}\n'.format(test_accuracy))
     f.write('TPrate: {:.3f}\n'.format(test_tp_rate))
     f.write('TNrate: {:.3f}\n'.format(test_tn_rate))
     f.write('TP adedi: {}\n'.format(test_tp_count))
     f.write('TN adedi: {}\n'.format(test_tn_count))




"""
- Using ID3 algorithm for Visualization of Decision Tree - 
I used the 2nd CART algorithm you see here(scroll down) with a 3rd party library only for image processing.
All of the classification and node rate operations that you have seen above, I extracted with the CART algorithm I wrote myself.
In the CART algorithm I wrote myself, I could not do image processing due to the fact that sklearn did not allow it. Otherwise, I would have written the image processing model myself.
"""
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

train_data = pd.read_csv("trainSet.csv")
test_data = pd.read_csv("testSet.csv")

categorical_columns = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]

le = LabelEncoder()
for column in categorical_columns:
    train_data[column] = le.fit_transform(train_data[column])
    test_data[column] = le.transform(test_data[column])

train_data['class'] = train_data['class'].replace({'good': 1, 'bad': 0})
test_data['class'] = test_data['class'].replace({'good': 1, 'bad': 0})

model = DecisionTreeClassifier()

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

model.fit(X_train, y_train)

plt.figure(figsize=(20, 20))
plot_tree(model, filled=True, feature_names=train_data.columns[:-1], class_names=['bad', 'good'], fontsize=14)
plt.savefig('decision_tree.png')
