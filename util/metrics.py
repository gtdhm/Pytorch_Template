# =====================util.metrics.py=========================
# This module contains useful evaluation metrics functions for
# directly Calculate the output accuracy of the network .
#
# Version: 1.0.0
# Date: 2019.05.20
# =============================================================

import math
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
warnings.simplefilter("ignore")
CONFIDENCES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


###############################################################
# Base Metrics class
###############################################################
class BaseMetrics(ABC):
    """This is an abstract base class (ABC), through inheriting it and rewriting the
    corresponding method to complete the evaluation metrics of the model.
    """
    def __init__(self, class_names):
        """Initialize common variables"""
        assert isinstance(class_names, list), "[Error] class_name should be a list!"
        self.count = 0
        self.y_true = self.y_pred = self.y_hat = 0
        self.metrics = {}
        self.class_names = class_names
        self.confusion_matrix = np.zeros((len(class_names), 4)).astype(int)

    @ abstractmethod
    def eval(self, y_true, y_pred, indicators="all", step=1):
        """Eval the label and the predict."""
        pass

    @ abstractmethod
    def cal_confusion_matrix(self):
        """Calculate binary class confusion matrix."""
        pass

    def report_template(self, message, name, length, metrics):
        """Fill the data in the report template."""
        end = ""
        title = " " * (length + 4) + "Accuracy  Precision  Recall   F1-score     FPR     Number"
        message += "\n" + " " * (length - len(name)) + name + " :"
        for i in range(0, 5):
            if math.isnan(metrics[i]):
                metrics[i] = "--"
                message += " " * (8 - len(metrics[i]) + 2) + metrics[i]
            else:
                if name == "TOTAL":
                    avg = len(self.class_names)
                else:
                    avg = 1.0
                message += " " * (8 - len("{:.3f}%".format(metrics[i] / avg * 100)) + 2) + \
                           "{:.3f}%".format(metrics[i] / avg * 100)
        message += "    {:d}".format(metrics[5])
        return title, message, end

    def plot_curve(self, confidences, save_path=None, show=True):
        """Plot and save PR Curve and ROC Curve."""
        precision = []
        recall = []
        fprs = []
        tprs = []
        for confidence in confidences:
            self.confusion_matrix[self.confusion_matrix >= 0] = 0
            self.y_hat = np.where(self.y_pred >= confidence, 1, 0)
            self.cal_confusion_matrix()
            _, p, r, _, tpr, fpr = self._cal_metrics(self.confusion_matrix[0][0], self.confusion_matrix[0][1],
                                                     self.confusion_matrix[0][2], self.confusion_matrix[0][3])
            precision.append(p)
            recall.append(r)
            tprs.append(tprs)
            fprs.append(fpr)
        self._plot_curve(fprs, tprs, 'ROC Curve', 'True Positive Rate (TPR)', 'False Positive Rate (FPR)',
                         save_path, show=show)
        self._plot_curve(recall, precision, "PR Curve", "Recall (R)", "Precision (P)", save_path, show=show)

    @ staticmethod
    def voc_ap(rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]
            # AP= AP1 + AP2+ AP3+ AP4
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    @ staticmethod
    def _convert_to_numpy(x):
        """Check and convert variable type to numpy array."""
        out = {list: lambda z: np.array(z.copy()), torch.Tensor: lambda z: z.data.numpy().copy()
               }.get(type(x), lambda z: z.copy()).__call__(x)
        assert isinstance(out, np.ndarray), "[Error] Inputs should be tensor or numpy array or list!"
        return out

    @ staticmethod
    def _convert_to_one_hot(np_input, num_label):
        """Transform row vector into one hot encoding.
        Inputs:
            np_input: numpy.shape(batch,), likes [1, 4, 2, 7, 4, 4]
            num_label: the number of labels
        Returns:
            np_one_hot: one hot encoding(numpy), likes shape(batch, num_label)
        """
        np_one_hot = np.eye(num_label)[np_input.reshape(-1)]
        return np_one_hot

    @ staticmethod
    def _cal_metrics(tp, fn, fp, tn):
        """Calculate metrics."""
        acc = (tp + tn) / (tp + fn + fp + tn)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        tpr = tp / (tp + fn)
        f1 = (2 * p * r) / (p + r)
        fpr = fp / (fp + tn)
        return acc, p, r, f1, tpr, fpr

    @ staticmethod
    def _cal_auc(prob, labels):
        """Calculate AUC metric"""
        f = list(zip(prob, labels))
        rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
        rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
        posNum = 0
        negNum = 0
        for i in range(len(labels)):
            if labels[i] == 1:
                posNum += 1
            else:
                negNum += 1
        if posNum * negNum == 0:
            return 0.0
        auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
        return auc

    @ staticmethod
    def _plot_curve(x, y, title, x_lab, y_lab, save_path=False, show=False):
        """Plot the Curve and save them.
        Inputs:
            x: x axis of the picture
            y: y axis of the picture
            title: the title of the picture
            x_lab: the x-axis describe
            y_lab: the y-axis describe
            save_path: the dir to save the picture
            show: whether show the picture on the screen(bool)
        """
        plt.title(title)
        plt.plot(x, y, 'k')
        plt.plot([(0, 0), (1, 1)], 'r--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.ylabel(x_lab)
        plt.xlabel(y_lab)
        if save_path is not False:
            plt.savefig(save_path)
        if show:
            plt.show()


###############################################################
# Binary Class Metrics class
###############################################################
class BinaryClassMetrics(BaseMetrics):
    """This class is used to evaluate the accuracy of binary-classification models.

    Example:
        <<< bc_metrics = BinaryClassinMetrics()
            bc_metrics.eval(label, out.cpu())
            bc_metrics.metrics
    """
    def __init__(self, class_names):
        super(BinaryClassMetrics, self).__init__(class_names)
        self.acc = self.p = self.r = self.f1 = self.tpr = self.fpr = self.auc = 0

    def eval(self, y_true, y_pred, indicators="all", step=1, confidence=0.5):
        """
        Inputs:
            y_true: Ground truth (correct) labels, like [0, 1, 1, 0]
            y_pred: Predicted labels (the out of nn.Sigmoid) shape(batch, 1)
            indicators: choose which metrics to calculate, likes "all" or "ACC, P, R"
            confidence: a threshold to decide True label and False label
        """
        if indicators == "all":
            indicators = ["ACC", "P", "PR", "R", "F1", "TPR", "FPR", "ROC", "AUC"]
        else:
            indicators = indicators.replace(" ", "").split(",")

        self.metrics = {}
        self.y_true = super()._convert_to_numpy(y_true)
        self.y_pred = np.squeeze(super()._convert_to_numpy(y_pred).T)
        self.y_hat = np.where(self.y_pred >= confidence, 1, 0)

        self.cal_confusion_matrix()
        if self.count == step:
            self.acc, self.p, self.r, self.f1, self.tpr, self.fpr = \
                super()._cal_metrics(self.confusion_matrix[0][0], self.confusion_matrix[0][1],
                                     self.confusion_matrix[0][2], self.confusion_matrix[0][3])
            self.auc = super()._cal_auc(self.y_pred, self.y_true)
            for indicator in indicators:
                self.metrics[indicator] = {"ACC": self.acc, "P": self.p, "R": self.r, "F1": self.f1,
                                           "TPR": self.tpr, "FPR": self.fpr, "AUC": self.auc}.get(indicator, "")
            if "PR" in indicators or "ROC" in indicators:
                super().plot_curve(confidences=CONFIDENCES)
            self.confusion_matrix[self.confusion_matrix >= 0] = 0
            self.count = 0

    def cal_confusion_matrix(self):
        # TP: predict and label both = 1
        self.confusion_matrix[0][0] += ((self.y_hat == 1) & (self.y_true == 1)).sum()
        # FN: predict = index and label = index
        self.confusion_matrix[0][1] += ((self.y_hat == 0) & (self.y_true == 1)).sum()
        # FP: predict = index and label = index
        self.confusion_matrix[0][2] += ((self.y_hat == 1) & (self.y_true == 0)).sum()
        # TN: predict and label both = index
        self.confusion_matrix[0][3] += ((self.y_hat == 0) & (self.y_true == 0)).sum()
        self.count += 1


###############################################################
# Multi Class Metrics class
###############################################################
class MultiClassMetrics(BaseMetrics):
    """This class is used to evaluate the accuracy of multi-classification models.
    Inputs:
        class_names: a list includes all class names

    Example:
        <<< mc_metrics = MultiClassMetrics()
            mc_metrics.eval(label, out.cpu(), average="macro")
            mc_metrics.metrics
    """
    def __init__(self, class_names):
        super(MultiClassMetrics, self).__init__(class_names)
        self.max_length = 5
        self.micro_acc = self.micro_p = self.micro_r = self.micro_f1 = self.micro_tpr = self.micro_fpr = 0

    def eval(self, y_true, y_pred, indicators="all", average="macro", step=1):
        """
        Inputs:
            y_true: Ground truth (correct) labels, like [1, 3, 5, 2]
            y_pred: Predicted labels (the out of nn.Linear or nn.Softmax) shape(batch, nclass)
            indicators: choose which metrics to calculate, likes "all" or "ACC, P, R"
            average: calculate mode includes "macro" or "micro"
            step: how many step to cache the confusion_matrix so that it can calculate the step*batch y_pred metrics
        """
        if indicators == "all":
            indicators = ["ACC", "P", "R", "F1", "TPR", "FPR", "Report"]
        else:
            indicators = indicators.replace(" ", "").split(",")

        self.macro_metrics = [[], [], [], [], [], []]
        self.metrics = {}

        self.y_true = super()._convert_to_numpy(y_true)
        self.y_pred = super()._convert_to_numpy(y_pred)
        self.y_hat = np.argmax(self.y_pred, axis=1)

        self.cal_confusion_matrix()
        if self.count == step:
            self.count = 0
            self.cal_micro_metrics()
            self.cal_macro_metrics()
            if "Report" in indicators:
                report = self.classification_report()
            else:
                report = "--"
            if average == "micro":
                for indicator in indicators:
                    self.metrics[indicator] = {"ACC": self.micro_acc, "P": self.micro_p, "R": self.micro_r,
                                               "F1": self.micro_f1, "TPR": self.micro_tpr, "FPR": self.micro_fpr,
                                               "Report": report}.get(indicator, 0.0)
            elif average == "macro":
                for indicator in indicators:
                    self.metrics[indicator] = {"ACC": self.micro_acc, "P": np.mean(self.macro_metrics[1]),
                                               "R": np.mean(self.macro_metrics[2]), "F1": np.mean(self.macro_metrics[3]),
                                               "TPR": np.mean(self.macro_metrics[4]), "FPR": np.mean(self.macro_metrics[5]),
                                               "Report": report}.get(indicator, "")
            else:
                raise IOError("[Error] param: average --> '{:s}' was not found!".format(average))
            self.confusion_matrix[self.confusion_matrix >= 0] = 0

    def cal_confusion_matrix(self):
        for index, name in enumerate(self.class_names):
            self.max_length = len(name) if len(name) > self.max_length else self.max_length
            # TP: predict and label both = index
            self.confusion_matrix[index][0] += ((self.y_hat == index) & (self.y_true == index)).sum()
            # FN: predict != index and label = index
            self.confusion_matrix[index][1] += ((self.y_hat != index) & (self.y_true == index)).sum()
            # FP: predict = index and label != index
            self.confusion_matrix[index][2] += ((self.y_hat == index) & (self.y_true != index)).sum()
            # TN: predict and label both != index
            self.confusion_matrix[index][3] += self.confusion_matrix[index][0]
        self.count += 1

    def cal_micro_metrics(self):
        """Calculate multi class micro-metrics."""
        # micro average
        matrix_sum = self.confusion_matrix.sum(axis=0)
        # calculate metrics
        self.micro_acc,  self.micro_p,  self.micro_r, self.micro_f1,  self.micro_tpr,  self.micro_fpr \
            = super()._cal_metrics(matrix_sum[0], matrix_sum[1], matrix_sum[2], matrix_sum[3])

    def cal_macro_metrics(self):
        """Calculate multi class macro-metrics."""
        lll = np.unique(np.append(self.y_hat, self.y_true))
        for index in lll:
            # calculate metrics
            acc, p, r, f1, tpr, fpr = super()._cal_metrics(self.confusion_matrix[index][0],
                                                           self.confusion_matrix[index][1],
                                                           self.confusion_matrix[index][2],
                                                           self.confusion_matrix[index][3])
            for i, metric in enumerate([acc, p, r, f1, tpr, fpr]):
                if not math.isnan(metric):
                    self.macro_metrics[i].append(metric)
                else:
                    self.macro_metrics[i].append(0.0)

    def classification_report(self):
        """Calculate multi class classification report."""
        message = ""
        total = [0, 0, 0, 0, 0, 0]
        for index, name in enumerate(self.class_names):
            # calculate metrics
            acc, p, r, f1, tpr, fpr = super()._cal_metrics(self.confusion_matrix[index][0],
                                                           self.confusion_matrix[index][1],
                                                           self.confusion_matrix[index][2],
                                                           self.confusion_matrix[index][3])
            number = self.confusion_matrix[index][0] + self.confusion_matrix[index][1]
            metrics = [acc, p, r, f1, fpr, number]
            for i, metric in enumerate(metrics):
                if math.isnan(metric):
                    metric = 0
                total[i] += metric
            title, message, end = super().report_template(message, name, self.max_length, metrics)
        title, message, end = super().report_template(message, "TOTAL", self.max_length, total)
        return title + message + end


###############################################################
# Multi Label Metrics class
###############################################################
class MultiLabelMetrics(BaseMetrics):
    """This class is used to evaluate the accuracy of multi-label classification models.

    Example:
        <<< ml_metrics = MultiLabelMetrics()
            ml_metrics.eval(label, out.cpu())
            ml_metrics.metrics
    """

    def __init__(self, class_names):
        super(MultiLabelMetrics, self).__init__(class_names)
        self.max_length = 5
        self.AP_metrics = []
        self.prec = np.zeros((len(self.class_names), len(CONFIDENCES))).astype(float)
        self.rec = np.zeros((len(self.class_names), len(CONFIDENCES))).astype(float)

    def eval(self, y_true, y_pred, indicators="all", step=1):
        """
        Inputs:
            y_true: Ground truth (correct) labels, like [[0, 1, 1], [1, 0, 1], ...]
            y_pred: Predicted labels (the out of nn.Sigmoid) shape(batch, nclass)
            indicators: choose which metrics to calculate, likes "all" or "ACC, P, R"
            confidence: a threshold to decide True label and False label
        """
        if indicators == "all":
            indicators = ["ACC", "P", "R", "F1", "TPR", "FPR"]
        else:
            indicators = indicators.replace(" ", "").split(",")

        self.metrics = {}
        self.mm = self.macro_metrics = [[], [], []]
        self.y_true = super()._convert_to_numpy(y_true)
        self.y_pred = super()._convert_to_numpy(y_pred)
        # print(self.y_true)
        # print(self.hhh)

        self.count += 1
        for ind1, confidence in enumerate(CONFIDENCES):
            self.confusion_matrix[self.confusion_matrix >= 0] = 0
            self.y_hat = np.where(self.y_pred >= confidence, 1, 0)
            self.cal_confusion_matrix()
            # calculate metrics
            for ind2, matrix in enumerate(self.confusion_matrix):
                acc, p, r, f1, tpr, fpr = super()._cal_metrics(matrix[0], matrix[1], matrix[2], matrix[3])
                if confidence == 0.5:
                    self.mm[0].append(acc)
                    self.mm[1].append(f1)
                    self.mm[2].append(fpr)
                if not math.isnan(p):
                    self.prec[ind2][ind1] = p
                else:
                    self.prec[ind2][ind1] = 1
                if not math.isnan(r):
                    self.rec[ind2][ind1] = r
                else:
                    self.rec[ind2][ind1] = 1
        # print(self.prec)
        # print(self.rec)
        for i in range(len(self.class_names)):
            self.AP_metrics.append(super().voc_ap(self.rec[i], self.prec[i]))
        # for i in range(len(self.AP_metrics) - 1, -1, -1):  # 从 len(e)-1 到 0删除3和4
        #     j = self.AP_metrics[i]
        #     if j == 0:
        #         self.AP_metrics.remove(self.AP_metrics[i])

        if self.count == step:
            for indicator in indicators:
                self.metrics[indicator] = {"mAP": np.mean(self.AP_metrics),
                                           "ACC": np.mean(self.mm[0]),
                                           "F1": np.mean(self.mm[1]),
                                           "FPR": np.mean(self.mm[2])}.get(indicator, "")
            self.count = 0
        self.prec[self.prec >= 0] = 0
        self.rec[self.rec >= 0] = 0
        # print((self.AP_metrics))
        # print(np.mean(self.AP_metrics))

    def cal_confusion_matrix(self):
        # for each batch
        for batch in range(len(self.y_hat)):
            for index in range(len(self.class_names)):
                self.confusion_matrix[index][0] += ((self.y_hat[batch][index]==1)&(self.y_true[batch][index]==1)).sum()
                self.confusion_matrix[index][1] += ((self.y_hat[batch][index]==0)&(self.y_true[batch][index]==1)).sum()
                self.confusion_matrix[index][2] += ((self.y_hat[batch][index]==1)&(self.y_true[batch][index]==0)).sum()
                self.confusion_matrix[index][3] += ((self.y_hat[batch][index]==0)&(self.y_true[batch][index]==0)).sum()

