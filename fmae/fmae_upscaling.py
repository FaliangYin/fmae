from fmae.fmae_base import FmaeBasicExplainer
from fmae.utilities import *
from sklearn.metrics import r2_score, accuracy_score
import sklearn

"""
FMAE for upscaling the explanation from domain level to universe level
FmaeDownscaling: corresponding to the experiment of Section V.D Case 3 Upscaling via aggregation of the paper

Please refer to the paper for more details:
"Hierarchical Fuzzy Model-Agnostic Explanation: Framework, Algorithms and Interface for XAI"
https://ieeexplore.ieee.org/document/10731553
Authors: Faliang Yin, Hak-Keung Lam, David Watson
"""


class FmaeUpscaling:
    """
    Providing the explanations of the hierarchical FMAE framework in a bottom-up manner (domain to universe).
    Parameters
    ----------
    model: closed box (black box) model to be explained
    instances: instances of interest to be explained
    mode: regression or classification
    num_sam: number of samples
    num_fuzzy_set: number of fuzzy set for each feature
    """
    def __init__(self, model, instances, mode='regression', num_sam=5000, num_fuzzy_set=3):
        self.model = model  # closed box (black box) model to be explained
        if str(type(instances)).endswith("pandas.core.frame.DataFrame'>"):  # transform to np data if input is pd data
            instances = instances.values
        self.mode = mode  # regression or classification
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(instances)  # learn the distribution of data for sampling
        self.num_sam = num_sam  # number of samples
        self.kernel_width = 0.5 * np.sqrt(instances.shape[-1])  # kernel width for sampling

        self.num_fea = instances.shape[1]  # number of input features
        self.num_class = 1  # number of output (classes)
        self.num_fuzzy_set = num_fuzzy_set  # number of fuzzy set for each feature
        self.score = None  # performance score for approximation ability of the FLS
        self.explainer = None  # basic TSK FLS model of FMAE
        self.explainer_list = []  # trained explainers for aggregation

    def explain(self, instances):
        if str(type(instances)).endswith("pandas.core.frame.DataFrame'>"):  # transform to np array if input is pd data
            instances = instances.values
        fs_all, score = [], []
        for instance in instances:
            # for each instance, generate samples, labels and weights to train an FLS as explainer
            samples = get_samples(instance, self.num_sam, self.scaler)
            labels = self.model(samples)
            if self.mode == 'regression':
                labels = np.expand_dims(labels, axis=1)
            else:
                self.num_class = labels.shape[1]
            self.explainer = FmaeBasicExplainer(self.num_fea, self.num_class, self.num_fuzzy_set, self.mode)
            weights = get_weights(samples, self.scaler, kernel_width=self.kernel_width)
            self.explainer.fit(samples, labels, sample_weight=weights)
            fs = self.explainer.coef_  # generate feature salience explanations
            fs_all.append(fs)
            score.append(self.explainer.score(samples, labels, sample_weight=weights))
            self.explainer_list.append(self.explainer.explainer)  # archive the explainers for aggregation
        self.score = np.average(score)
        return np.array(fs_all)

    def weight_aggregation(self, x):
        # aggregate the archived explainers by weight aggregation, please see Algorithm 1 in paper for more details
        if str(type(x)).endswith("pandas.core.frame.DataFrame'>"):  # transform to np data if input is pd data
            x = x.values
        fs_all, score_all = [], []
        for i in range(x.shape[0]):
            # calculate the predictions by the explainers to be aggregated
            valid_set = get_samples(x[i], self.num_sam, self.scaler)
            valid_labels = self.model(valid_set)
            if self.mode == 'regression':
                valid_labels = np.expand_dims(valid_labels, axis=1)
            sample_weights = get_weights(valid_set, self.scaler, kernel_width=self.kernel_width, flag=True)
            predictions, fss = [], []
            score_0 = []
            for explainer in self.explainer_list:
                score_0.append(explainer.score(valid_set, valid_labels, sample_weight=sample_weights))
                if self.mode == 'regression':
                    prediction = np.squeeze(explainer.predict(valid_set), axis=1)
                else:
                    prediction = explainer.predict(valid_set)

                predictions.append(prediction)
                fs = self.explainer.feature_attribution(np.expand_dims(x[i], axis=0))
                fss.append(fs)
            predictions, fss = np.array(predictions), np.array(fss)

            # calculate the weights (aggregation membership) for aggregation
            if self.mode == 'classification':
                aggregation_weights = []
                fs_agg, prediction = [], []
                for j in range(self.num_class):
                    system_weight = np.linalg.inv(predictions[:, :, j] @ np.diag(sample_weights) @ predictions[:, :, j].T #
                                                  + 0.1 * np.eye(predictions.shape[0])) @ predictions[:, :, j] \
                                                 @ np.diag(sample_weights) @ valid_labels[:, j]
                    fs_agg.append(system_weight @ fss[:, :, j])
                    prediction.append(system_weight @ predictions[:, :, j])
                    aggregation_weights.append(system_weight)
                fs_agg, prediction = np.array(fs_agg).T, np.array(prediction).T
                score = accuracy_score(valid_labels.argmax(axis=1), prediction.argmax(axis=1), sample_weight=sample_weights)
            else:
                predictions = predictions.T
                aggregation_weights = np.linalg.inv(predictions.T @ np.diag(sample_weights) @ predictions +  #
                                                    0.1 * np.eye(predictions.shape[1])) @ predictions.T \
                                      @ np.diag(sample_weights) @ valid_labels[:, 0]
                fs_agg = aggregation_weights @ fss
                prediction = predictions @ aggregation_weights
                score = r2_score(valid_labels, prediction, sample_weight=sample_weights)

            fs_all.append(fs_agg)
            score_all.append(score)
        return np.array(fs_all), np.average(score_all)
