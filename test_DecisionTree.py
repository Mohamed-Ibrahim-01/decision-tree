import unittest
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from pandas.testing import assert_frame_equal
from DecisionTree import *

class TestDecisionTree(unittest.TestCase):

    def setUp(self): 
        x = {"A":[1, 1, 1, 1, 1, 1, 1, 1],\
             "B":[0, 0, 0, 0, 1, 1, 1, 1]}
        y = {"label": [0, 0, 1, 1, 1, 1, 1, 1]}
        self.X = pd.DataFrame(x)
        self.Y = pd.DataFrame(y)
        self.dataset = pd.DataFrame({"A":[1, 1, 1, 1, 1, 1, 1, 1],\
                                     "B":[0, 0, 0, 0, 1, 1, 1, 1],\
                                     "label": [0, 0, 1, 1, 1, 1, 1, 1]})
        self.dataset_features_values = {"A":{1:8},"B":{0:4, 1:4}, "label": {0:2, 1:6}}
        self._dt = DecisionTree()
        self._dt.features_values = self.dataset_features_values

    def test_build_records(self):
        builded_records = build_records(self.X, self.Y)
        assert_frame_equal(self.dataset, builded_records)

    def test_attribute_value(self):
        sample = self.dataset.iloc[0]
        self.assertEqual(attribute_value(sample, "label"), 0)
        self.assertEqual(attribute_value(sample, "A"), 1)
        self.assertEqual(attribute_value(sample, "B"), 0)
        with self.assertRaises(KeyError):
            attribute_value(sample, "garbage")

    def test_get_feature_values(self):
        a_values = get_feature_values(self.dataset, "A")
        b_values = get_feature_values(self.dataset, "B")

        self.assertEqual(a_values, {1: 8})
        self.assertEqual(b_values, {0: 4, 1: 4})

    def test_calc_entropy(self):
        entropy = self._dt._calc_entropy(self.dataset, "label")
        self.assertAlmostEqual(entropy, 0.811278, places=5)

    def test_calc_conditional_entropy(self):
        a_conditional_entropy = self._dt._calc_weighted_entropy(self.dataset, "A")
        b_conditional_entropy = self._dt._calc_weighted_entropy(self.dataset, "B")
        self.assertAlmostEqual(a_conditional_entropy, 0.811278, places=5)
        self.assertAlmostEqual(b_conditional_entropy, 0.5, places=5)

    def test_information_gain(self):
        a_information_gain = self._dt._information_gain(self.dataset, "A")
        b_information_gain = self._dt._information_gain(self.dataset, "B")
        self.assertAlmostEqual(a_information_gain, 0.0, places=5)
        self.assertAlmostEqual(b_information_gain, 0.311278, places=5)

    def test_get_split_feature(self):
        split_feature = self._dt._get_split_feature(self.dataset)
        self.assertEqual(split_feature, "B")

    def test_split_by(self):
        splits = self._dt._split_by(self.dataset, "B")
        b_split_0 = pd.DataFrame({"A":[1, 1, 1, 1], "B":[0, 0, 0, 0],\
                "label": [0, 0, 1, 1]}, index=[0, 1, 2, 3] )
        b_split_1 = pd.DataFrame({"A":[1, 1, 1, 1], "B":[1, 1, 1, 1],\
                "label": [1, 1, 1, 1]}, index=[4, 5, 6 , 7] )

        for value, split_data in splits:
            if value == 0: assert_frame_equal(split_data, b_split_0)
            if value == 1: assert_frame_equal(split_data, b_split_1)

    def test_tree_node_ctor(self):
        node = TreeNode(self.dataset, "")

        assert_frame_equal(node.data, self.dataset)
        self.assertEqual(node.split_feature, "")
        self.assertEqual(node.branches, dict())

    def test_node_tree_equality_overload(self):
        pure_data_set = self.dataset[self.dataset["label"] == 1]
        node1 = TreeNode(self.dataset.copy())
        node0 = TreeNode(self.dataset.copy())
        node2 = TreeNode(pure_data_set)

        self.assertEqual(node0, node1)

        node0.split_feature = "A"
        self.assertNotEqual(node0, node1)
        self.assertNotEqual(node0, node2)
        self.assertNotEqual(node1, node2)


    def test_node_tree_classify(self):
        pure_data_set = self.dataset[self.dataset["label"] == 1]
        node = TreeNode(pure_data_set, "NO_SPLIT")

        tree = TreeNode(self.dataset, "B")
        branches = {0:TreeNode(self.dataset[self.dataset["B"] == 0], "NO_SPLIT"),\
                    1:TreeNode(self.dataset[self.dataset["B"] == 1], "NO_SPLIT")}
        tree.branches = branches

        self.assertEqual(node.classify(self.dataset.iloc[3]), 1)

        expected = [0, 0, 0, 0, 1, 1, 1, 1]
        for i in range(8):
            self.assertEqual(tree.classify(self.dataset.iloc[i]), expected[i])

    def test_DecisionTree_build(self):
        dt = DecisionTree()
        dt.features_values = self.dataset_features_values
        root_split = dt._get_split_feature(self.dataset)
        root = TreeNode(self.dataset, root_split)
        tree = dt._build_tree(root)

        expexted_tree = TreeNode(self.dataset,"B")
        branches = {0:TreeNode(self.dataset[self.dataset["B"] == 0],"NO_SPLIT"),\
                    1:TreeNode(self.dataset[self.dataset["B"] == 1],"NO_SPLIT")}
        expexted_tree.branches = branches

        self.assertEqual(tree, expexted_tree)

    def test_DecisionTree_fit(self):
        dt = DecisionTree()
        dt.features_values = self.dataset_features_values
        dt.fit(self.X, self.Y)

        expexted_tree = TreeNode(self.dataset, "B")
        branches = {0:TreeNode(self.dataset[self.dataset["B"] == 0],"NO_SPLIT"),\
                    1:TreeNode(self.dataset[self.dataset["B"] == 1],"NO_SPLIT")}
        expexted_tree.branches = branches

        self.assertEqual(dt.tree, expexted_tree)

    def test_DecisionTree_predict(self):
        dt = DecisionTree()
        dt.features_values = self.dataset_features_values
        dt.fit(self.X, self.Y)
        predicted = dt.predict(self.X)
        expected_predictions = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(expected_predictions, predicted)

    def compare_sklearn_mytree(self, X, Y):
        sk_dt = DecisionTreeClassifier(criterion='entropy')
        sk_dt.fit(X, Y)
        sk_predicted = sk_dt.predict(X)

        my_dt = DecisionTree(criterion='entropy')
        my_dt.fit(X, Y)
        my_predicted = my_dt.predict(X)

        np.testing.assert_array_equal(my_predicted, sk_predicted)

    def test_compare_to_sklearn(self):
        students_dataset = pd.read_csv("students.csv")
        cardio_dataset = pd.read_csv("cardio_train.csv", sep = ';')
        if(isinstance(students_dataset, pd.DataFrame) and isinstance(cardio_dataset, pd.DataFrame)):
            students_X = students_dataset.drop(['label'], axis = 1)
            students_Y = students_dataset["label"]

            cardio_dataset = cardio_dataset.iloc[:,:]
            cardio_dataset.rename(columns = {'cardio':'label'}, inplace = True)
            cardio_dataset.rename(columns = {'cardio':'label'}, inplace = True)
            cardio_dataset.drop(['id', 'ap_hi', 'ap_lo', 'height', 'weight', 'age'], axis = 1, inplace=True)

            cardio_X = cardio_dataset.drop(['label'], axis = 1)
            cardio_Y = cardio_dataset['label']

            self.compare_sklearn_mytree(self.X, self.Y)
            self.compare_sklearn_mytree(students_X, students_Y)
            self.compare_sklearn_mytree(cardio_X, cardio_Y)
        else: raise RuntimeError("datasets not loaded correctrly")


if __name__ == '__main__':
    unittest.main()

