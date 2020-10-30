from abc import abstractmethod

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, auc
import numpy as np


class Model:
    """
    Abstract class to define which model will serve as a classifier for data.
    """

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class Metrics:
    """
    Implements Metrics Class for classification evaluation.
    """

    def accuracy(self, y_true, y_pred):
        """
        Returns accuracy score metric
        :param y_true:
        :param y_pred:
        :return: acc
        """
        return accuracy_score(y_true, y_pred)

    def cnf_mtx(self, y_true, y_pred):
        """
        Determines the confusion matrix for a given prediction.
        :param y_true:
        :param y_pred:
        :return: confusion_matrix
        """
        return confusion_matrix(y_true, y_pred)

    def report(self, y_true, y_pred):
        """
        Give Classification Report
        :param y_true:
        :param y_pred:
        :return:
        """
        return classification_report(y_true, y_pred)

    def apfd(self, prioritization: list):
        """
        Calculates the Average Percentage of Fault Detection. Given a prioritiation the APFD is near 1, if all relevant
        tests are applied at the beginning and 0 otherwise.
        :param prioritization:
        :return: apfd
        """
        n = len(prioritization)
        m = sum(prioritization)

        pos = 0
        if m != 0:
            for i in range(n):
                if prioritization[i] == 1:
                    pos += i
            return 1 - pos / (n * m) + 1 / (2 * n)
        else:
            return None

    def fdr(self, y_pred, duration):
        """
        Fault detection rate: y-axis -> percentage of faults detected - x-axis -> percentage of time spent
        :param duration:
        :param y_pred:
        :return: fdr
        """
        faults = []
        time = []
        count_f = 0
        count_d = 0
        for idx, i in enumerate(y_pred):
            if i == 1:
                count_f += 1
            count_d += duration[idx]

            faults.append(count_f)
            time.append(count_d)
        if max(faults) == 0:
            return 0.5
        else:
            return np.round(auc(x=np.array(time) / max(time), y=np.array(faults) / max(faults)), 2)

    def pretty_print_stats(self):
        """
        returns report of statistics for a given model object
        """
        items = (('accuracy:', self.accuracy()), ('sst:',),
                 ('mse:',), ('r^2:',),
                 ('adj_r^2:',))

        for item in items:
            print('{0:8} {1:.4f}'.format(item[0], item[1]))
