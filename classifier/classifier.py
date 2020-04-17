from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score

from compound.compound import Compound


class Classifier:
    def __init__(self, train_compounds: List[Compound], validation_compounds: List[Compound]):
        self.train_compounds = train_compounds
        self.validation_compounds = validation_compounds
        self.classifier = LogisticRegression(random_state=0)

    def calculate_accuracy(self, available_features: List[int]) -> float:
        x_train, y_train, x_validation, y_validation = self._train_validation_split(available_features)

        self._train(x_train, y_train)

        predicted_result = self.classifier.predict(x_validation)

        cm = confusion_matrix(y_validation, predicted_result)
        correct_predictions = cm[0, 0] + cm[1, 1]

        return correct_predictions / len(y_validation)

    def calculate_precision_recall(self, available_features: List[int]) -> float:
        x_train, y_train, x_validation, y_validation = self._train_validation_split(available_features)

        self._train(x_train, y_train)

        y_score = self.classifier.decision_function(x_validation)

        return average_precision_score(y_validation, y_score)

    def calculate_area_under_roc(self, available_features: List[int]) -> float:
        x_train, y_train, x_validation, y_validation = self._train_validation_split(available_features)

        self._train(x_train, y_train)

        probabilities = self.classifier.predict_proba(x_validation)[:, 1]
        area_under_curve = roc_auc_score(y_validation, probabilities)

        return area_under_curve

    def predict_result(self, test_compounds: List[Compound], available_features: List[int]) -> List[int]:
        x_train, y_train, x_to_predict, _ = self._train_validation_split(available_features,
                                                                         self.train_compounds + self.validation_compounds,
                                                                         test_compounds)

        self._train(x_train, y_train)

        return self.classifier.predict(x_to_predict)

    def _train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.classifier = LogisticRegression(random_state=0)
        self.classifier.fit(x_train, y_train)

    def _train_validation_split(self, available_features: List[int], train_compounds: List[Compound] = None,
                                validation_compounds: List[Compound] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train = train_compounds if train_compounds is not None else self.train_compounds
        validation = validation_compounds if validation_compounds is not None else self.validation_compounds

        x_train = compounds_to_features_array(train, available_features)
        y_train = compounds_to_active_array(train)
        x_validation = compounds_to_features_array(validation, available_features)
        y_validation = compounds_to_active_array(validation)

        return x_train, y_train, x_validation, y_validation


def compound_to_feature_array(compound: Compound, available_features: List[int]) -> List[int]:
    return [(1 if compound.has_feature(feature) else 0) for feature in available_features]


def compounds_to_features_array(compounds: List[Compound], available_features: List[int]) -> np.ndarray:
    return np.array([compound_to_feature_array(compound, available_features) for compound in compounds])


def compounds_to_active_array(compounds: List[Compound]) -> np.ndarray:
    return np.array([compound.active for compound in compounds])
