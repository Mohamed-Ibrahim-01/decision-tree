import numpy as np
import numpy as math
import pandas as pd


def build_records(X, y):
    """Bundels training data with labels in one dataframe to train with it

    Parameters
    ----------
    X : pd.DataFrame
        The training features only

    y : pd.DataFrame or pd.Series
        The labels of the training data

    Returns
    -------
    list
        one pd.DataFrame bundeled with both training features and labels
    """

    labeled_samples = []
    if isinstance(X, pd.DataFrame) and isinstance(y, (pd.DataFrame, pd.Series)):
        labeled_samples = X.copy()
        labeled_samples["label"] = y
    else:
        raise TypeError(f"""Unsupported data type of X :{type(X)} or Y : {type(y)}
                consider converting it to pd.DataFrame""")
    return labeled_samples


def attribute_value(sample, feature):
    """A wrapper method to access a sample feature

    Parameters
    ----------
    sample : pd.Series
        The sample to access

    feature : str
        The feature name

    Returns
    -------
    type(sample[feature])
        sample feature value is returned if found otherwise error is raised
    """
    if isinstance(sample, pd.Series):
        return sample[feature]
    else: 
        raise TypeError(f"""Unsupported sample type {type(sample)}
                consider converting it to pd.Series""")


def dataset_features_values(dataset):
    """A dictionary of all features values counts is formed form the whole dataset
    for exmaple {
                    "A":{"a":2, "b":5},
                    "B":{"a":3, "c":15},
                    ...
                }
    so the dataset has 2 features (A, B). The values of feature A are (a, b)
    and their corresponding values are (2, 5), and similarly for B.

    Parameters
    ----------
    sample : pd.DataFrame
        The dataset to get its features values count

    Returns
    -------
    dict(str: dict(str: int))
        dictionary of dictionary of values count
    """
    if isinstance(dataset, pd.DataFrame):
        features_values = dict()
        for feature in list(dataset):
            features_values[feature] = get_feature_values(dataset, feature)
        return features_values
    else: 
        raise TypeError(f"Unsupported sample type {type(dataset)}\
                consider converting it to pd.DataFrame")


def get_feature_values(data, feature):
    """A dictionary of feature values counts is formed form the whole dataset
    feature column for exmaple {"a":2, "b":5}. so the dataset has feature column
    with possible values (a, b) with number of occurrences (2, 5) respectivly

    Parameters
    ----------
    sample : pd.DataFrame
        The dataset to get its feature values count

    Returns
    -------
    dict(str: int)
        dictionary of values count ( number of occurrences ) of feature column in dataset
    """
    if isinstance(data, pd.DataFrame):
        if feature not in list(data): return dict()
        return data[feature].value_counts().to_dict()
    else: 
        raise TypeError(f"""Unsupported sample type {type(data)}
                consider converting it to pd.DataFrame""")


def majority_vote(dataset):
    """For a given dataset with label column the label value that appears the 
    most is returned

    Parameters
    ----------
    sample : pd.DataFrame
        The dataset with label column to apply vote on its labels occurrences

    Returns
    -------
    type(label)
        label that appears the most in label column if exist otherwise error is raised
    """
    if isinstance(dataset, pd.DataFrame):
        feature_values = get_feature_values(dataset, "label")
        feature_values_zip = zip(feature_values.values(), feature_values.keys())
        return sorted(feature_values_zip, key = lambda x: (-x[0], x[1]))[0][1]
    else: 
        raise TypeError(f"""Unsupported dataset type {type(dataset)}
                consider converting it to pd.DataFrame""")


class TreeNode:
    """The base block in the tree construction
    """

    def __init__(self, data, split_feature = ""):
        self.data = data
        self.split_feature = split_feature
        self.branches = dict()

    def classify(self, sample):
        """For a given sample return the label of it 

        Parameters
        ----------
        sample : pd.Series
            The sample needed to classify

        Returns
        -------
        type(label)
            label that appears the most in label column if exist otherwise error is raised
        """
        if self.split_feature =="NO_SPLIT": 
            return majority_vote(self.data)
        value = attribute_value(sample, self.split_feature)
        return self.branches[value].classify(sample)

    def __eq__(self, other): 
        if not isinstance(other, TreeNode):
            return NotImplemented

        return self.data.equals(other.data) and\
               self.branches == other.branches and\
               self.split_feature == other.split_feature

    def __repr__(self) -> str:
        tree_repr = ""
        tree_repr += f"**{self.split_feature}/{len(self.branches)}**"
        return tree_repr


class DecisionTree:
    def __init__(self, criterion='entropy'):
        self.criterion = criterion
        self.splitter = criterion
        self.tree = TreeNode(pd.DataFrame([]))
        self.features_values = dict()

    def fit(self, X, y):
        """Training on the given data

        Parameters
        ----------
        X : pd.DataFrame
            Training data without labels

        y : pd.DataFrame
            Training data labels

        Returns
        -------
        void
        """
        dataset = build_records(X, y)
        self.features_values = dataset_features_values(dataset)
        root_split = self._get_split_feature(dataset, self.criterion)
        root = TreeNode(dataset, root_split)
        tree = self._build_tree(root)
        self.tree = tree

    def predict(self, X):
        """Testing data to predict its labels

        Parameters
        ----------
        X : pd.DataFrame
            Training data without labels

        Returns
        -------
        array of labels
        """
        if isinstance(X, pd.DataFrame):
            predicted = []
            for _, sample in X.iterrows():
                predicted.append(self.tree.classify(sample))
            return np.array(predicted)
        else: 
            raise TypeError(f"Unsupported dataset type {type(X)}\
                    consider converting it to pd.DataFrame")

    def _build_tree(self, node):
        """Recursively building tree nodes using information gain for a given root(node)

        Parameters
        ----------
        node : TreeNode
            The root of the tree.

        Returns
        -------
        void
        """
        if node.split_feature == "NO_SPLIT": return node
        splited_data = self._split_by(node.data, node.split_feature)
        branches = dict()
        for feature_value, data in splited_data:
            branch_split_feature = self._get_split_feature(data, self.criterion)
            branch_node = TreeNode(data, branch_split_feature)
            self._build_tree(branch_node)
            branches[feature_value] = branch_node
        node.branches = branches
        return node

    def _calc_entropy(self, data, attribute = "label"):
        """Entropy claculations of a given dataset using attribute column (label is the default)
        attribute column

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to claculate entropy on

        attribute(optional) : 
            The column name to use in entropy claculations

        Returns
        -------
        int
            entropy of the dataset
        """
        entropy = 0
        if len(data) == 0: return 0
        attribute_values = set(self.features_values[attribute].keys())
        for value in attribute_values:
            label_probability = len(data[data[attribute] == value])/len(data)
            if(label_probability == 0): continue
            entropy -= label_probability * math.log2(label_probability)
        return entropy

    def _calc_weighted_entropy(self, data, attribute):
        """The weighted entropy of the data set after splitting with attribute column
        categories.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset with labels (label column included) to claculate entropy on

        attribute : 
            The feature(column) name to used to split the data depending on its categories
            (feature values)

        Returns
        -------
        int
            entropy of the dataset
        """
        conditional_entropy = 0
        attribute_values = set(self.features_values[attribute].keys())
        for value in attribute_values:
            conditional_data = data[data[attribute] == value]
            conditional_probability = len(conditional_data)/len(data)
            conditional_entropy += conditional_probability * self._calc_entropy(conditional_data)
        return conditional_entropy

    def _information_gain(self, dataset ,feature):
        """A spliting criterion used to calculate how good is feature to split dataset using entropy.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset with labels (label column included).

        feature : 
            The feature(column) name to used to split the dataset depending on its categories
            (feature values)

        Returns
        -------
        int
            how much information we gain from spliting using feature on dataset
        """
        entropy = self._calc_entropy(dataset)
        conditional_entropy = self._calc_weighted_entropy(dataset, feature)
        mutual_information = entropy - conditional_entropy
        return mutual_information

    def _get_split_feature(self, dataset, criterion = "entropy"):
        """A helper method to return the best feature to split on for a given dataset.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset with labels (label column included).

        criterion : str
            The method used to evalute the split.

        Returns
        -------
        str
            feature name
        """
        if criterion != "entropy":
            raise ValueError(f"Unsupported criterion {criterion} only 'entropy' supported")
        if isinstance(dataset, pd.DataFrame):
            split_feature, max_gain = "", 0
            features = list(dataset)
            features.remove("label")
            for feature in features:
                feature_gain = self._information_gain(dataset, feature)
                if feature_gain >= max_gain:
                    split_feature = feature
                    max_gain = feature_gain
            return split_feature if max_gain > 0 else "NO_SPLIT"

        else: 
            raise TypeError(f"""Unsupported dataset type {type(dataset)}
                    consider converting it to pd.DataFrame""")


    def _split_by(self, dataset, split_feature):
        """A helper method to split dataset with "split_feature"

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset with labels (label column included).

        split_feature : str
            The feature name to split data on.

        Returns
        -------
        list(tuples)
            list of tuple ("feature value", splited data)
        """
        if isinstance(dataset, pd.DataFrame):
            feature_values = set(self.features_values[split_feature].keys())
            splits = []
            for value in feature_values:
                splited_data = dataset[dataset[split_feature] == value]
                if(len(splited_data) > 0): splits.append((value, splited_data))
            return splits
        else: 
            raise TypeError(f"""Unsupported dataset type {type(dataset)}
                    consider converting it to pd.DataFrame""")


