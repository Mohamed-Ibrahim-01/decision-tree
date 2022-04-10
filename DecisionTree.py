import numpy as np
import numpy as math
import pandas as pd


def build_records(X, y):
    labeled_samples = []
    if isinstance(X, pd.DataFrame) and isinstance(y, (pd.DataFrame, pd.Series)):
        labeled_samples = X.copy()
        labeled_samples["label"] = y
    else:
        raise TypeError(f"""Unsupported data type of X :{type(X)} or Y : {type(y)}
                consider converting it to pd.DataFrame""")
    return labeled_samples


def attribute_value(sample, feature):
    if isinstance(sample, pd.Series):
        return sample[feature]
    else: 
        raise TypeError(f"Unsupported sample type {type(sample)}\
                consider converting it to pd.Series")


def dataset_features_values(dataset):
    if isinstance(dataset, pd.DataFrame):
        features_values = dict()
        for feature in list(dataset):
            features_values[feature] = get_feature_values(dataset, feature)
        return features_values
    else: 
        raise TypeError(f"Unsupported sample type {type(dataset)}\
                consider converting it to pd.DataFrame")

def get_feature_values(data, feature):
    if isinstance(data, pd.DataFrame):
        if feature not in list(data): return dict()
        return data[feature].value_counts().to_dict()
    else: 
        raise TypeError(f"Unsupported sample type {type(data)}\
                consider converting it to pd.DataFrame")


def calc_entropy(data, features_values, attribute = "label"):
    entropy = 0
    if len(data) == 0: return 0
    attribute_values = set(features_values[attribute].keys())
    for value in attribute_values:
        label_probability = len(data[data[attribute] == value])/len(data)
        if(label_probability == 0): continue
        entropy -= label_probability * math.log2(label_probability)
    return entropy


def calc_conditional_entropy(data, features_values, attribute):
    conditional_entropy = 0
    attribute_values = set(features_values[attribute].keys())
    for value in attribute_values:
        conditional_data = data[data[attribute] == value]
        conditional_probability = len(conditional_data)/len(data)
        conditional_entropy += conditional_probability * calc_entropy(conditional_data, features_values)
    return conditional_entropy


def information_gain(dataset, features_values ,feature):
    entropy = calc_entropy(dataset, features_values)
    conditional_entropy = calc_conditional_entropy(dataset, features_values, feature)
    mutual_information = entropy - conditional_entropy
    return mutual_information

def get_split_feature(dataset, features_values, criterion = "entropy"):
    if criterion != "entropy":
        raise ValueError(f"Unsupported criterion {criterion} only 'entropy' supported")
    if isinstance(dataset, pd.DataFrame):
        split_feature, max_gain = "", 0
        features = list(dataset)
        features.remove("label")
        for feature in features:
            feature_gain = information_gain(dataset, features_values, feature)
            if feature_gain >= max_gain:
                split_feature = feature
                max_gain = feature_gain
        return split_feature if max_gain > 0 else "NO_SPLIT"

    else: 
        raise TypeError(f"Unsupported dataset type {type(dataset)}\
                consider converting it to pd.DataFrame")


def split_by(dataset, features_values, split_feature):
    if isinstance(dataset, pd.DataFrame):
        feature_values = set(features_values[split_feature].keys())
        splits = []
        for value in feature_values:
            splited_data = dataset[dataset[split_feature] == value]
            if(len(splited_data) > 0): splits.append((value, splited_data))
        return splits
    else: 
        raise TypeError(f"Unsupported dataset type {type(dataset)}\
                consider converting it to pd.DataFrame")

def majority_vote(dataset):
    if isinstance(dataset, pd.DataFrame):
        feature_values = get_feature_values(dataset, "label")
        feature_values_zip = zip(feature_values.values(), feature_values.keys())
        return sorted(feature_values_zip, key = lambda x: (-x[0], x[1]))[0][1]
    else: 
        raise TypeError(f"Unsupported dataset type {type(dataset)}\
                consider converting it to pd.DataFrame")

class TreeNode:
    def __init__(self, data, features_values, split_feature = ""):
        self.data = data
        self.split_feature = split_feature
        self.branches = dict()
        self.features_values = features_values
        self.is_pure = self._is_pure()

    def classify(self, sample):
        if self.is_pure or self.split_feature =="NO_SPLIT": 
            return majority_vote(self.data)
        value = attribute_value(sample, self.split_feature)
        return self.branches[value].classify(sample)

    def _is_pure(self):
        return len(set(get_feature_values(self.data, "label").keys())) == 1

    def __eq__(self, other): 
        if not isinstance(other, TreeNode):
            return NotImplemented

        return self.data.equals(other.data) and\
               self.branches == other.branches and\
               self.is_pure == other.is_pure and\
               self.split_feature == other.split_feature

    def __repr__(self) -> str:
        tree_repr = ""
        tree_repr += f"**{self.split_feature}/{self.is_pure}/{len(self.branches)}**"
        return tree_repr


class DecisionTree:
    def __init__(self, criterion='entropy'):
        self.criterion = criterion
        self.splitter = criterion
        self.tree = TreeNode(pd.DataFrame([]), dict())
        self.features_values = dict()

    def fit(self, X, y):
        dataset = build_records(X, y)
        self.features_values = dataset_features_values(dataset)
        root_split = get_split_feature(dataset, self.features_values ,self.criterion)
        root = TreeNode(dataset, self.features_values,root_split)
        tree = self._build_tree(root)
        self.tree = tree

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            predicted = []
            for _, sample in X.iterrows():
                predicted.append(self.tree.classify(sample))
            return np.array(predicted)
        else: 
            raise TypeError(f"Unsupported dataset type {type(X)}\
                    consider converting it to pd.DataFrame")

    def _build_tree(self, node):
        if node.is_pure or node.split_feature == "NO_SPLIT": return node
        splited_data = split_by(node.data, node.features_values, node.split_feature)
        branches = dict()
        for feature_value, data in splited_data:
            branch_split_feature = get_split_feature(data, node.features_values, self.criterion)
            branch_node = TreeNode(data, node.features_values, branch_split_feature)
            self._build_tree(branch_node)
            branches[feature_value] = branch_node
        node.branches = branches
        return node


