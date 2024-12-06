import numpy as np
from sklearn.linear_model import Ridge
import copy
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from functools import partial
from sklearn import model_selection
from sklearn.metrics.pairwise import pairwise_distances

"""
Library of the functions used in FMAE

Please refer to the paper for more details:
"Hierarchical Fuzzy Model-Agnostic Explanation: Framework, Algorithms and Interface for XAI"
https://ieeexplore.ieee.org/document/10731553
Authors: Faliang Yin, Hak-Keung Lam, David Watson
"""

def get_weights(scaled_data, scaler, kernel_width=0.5, flag=True):
    # get weights for each samples with the value inversely proportional to the distance from the original instance
    # this function derive from LIME [3]
    def kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    if flag is True:
        scaled_data = (scaled_data - scaler.mean_) / scaler.scale_
        kernel_fn = partial(kernel, kernel_width=kernel_width)
        distances = pairwise_distances(
            scaled_data,
            scaled_data[0].reshape(1, -1),
            metric='euclidean'
        ).ravel()
        return kernel_fn(distances)
    else:
        return None


def get_samples(instance, num_samples, scaler, sample_around_instance=True, zoomer=1, standardization=False):
    # sampling around the instance to train the explainer
    num_cols = instance.shape[0]
    instance_sample = instance
    scale = scaler.scale_
    mean = scaler.mean_
    random_state = check_random_state(None)
    data = random_state.normal(
        0, 1, num_samples * num_cols).reshape(
        num_samples, num_cols)
    if sample_around_instance:
        data = data * scale * zoomer + instance_sample
    else:
        data = data * scale * zoomer + mean
    data[0] = instance.copy()
    if not standardization:
        return data
    else:
        return (data - scaler.mean_) / scaler.scale_


def calculate_error(samples, responses, explainer, weight=None):
    # calculate mse error of explainer
    prediction, fir_str_bar, membership_value = explainer.forward(samples)
    loss = mean_squared_error(responses, prediction, sample_weight=weight)
    return loss, fir_str_bar, membership_value


def fast_update_error(rem_rule_output, rem_fir_str_bar, responses, prediction, weight=None):
    # a fast update of the error when removing a rule
    removements = np.einsum('NC,N->NC', rem_rule_output, rem_fir_str_bar)
    prediction = prediction - removements
    loss = mean_squared_error(responses, prediction, sample_weight=weight)
    return loss


def tune_model(explainer, sam_train, sam_labels_train, sample_weight=None):
    # fine-tune the consequent parameters of FLS
    explainer.lse_con_param(sam_train, sam_labels_train, sample_weight)
    tra_loss, fir_str_bar, membership_value = calculate_error(sam_train, sam_labels_train, explainer, sample_weight)
    return explainer, tra_loss, fir_str_bar


def rule_reduction(explainer, fir_str_bar, sam_train, sam_labels_train, tra_loss, tra_loss0, tau_R,
                        log=False, weight=None):
    # simplify the rule base by removing redundant rules, please see Algorithm 2 in paper for more details
    # get the rule removing order list
    sort_id = np.argsort(np.mean(fir_str_bar, axis=0), axis=0)[::-1]  # from high to low
    explainer.con_param = explainer.con_param[:, sort_id, :]
    explainer.rule_base = explainer.rule_base[sort_id, :]
    fir_str_bar = fir_str_bar[:, sort_id]

    i_max = explainer.num_rule-1
    MR_his = [tra_loss]
    # calculate the initial prediction
    rule_output = explainer.consequent(sam_train)
    prediction = np.einsum('NRC,NR->NC', rule_output, fir_str_bar)
    avg_tra_loss, i = tra_loss, 0
    explainer_copy = copy.deepcopy(explainer)
    while i <= i_max:
        if i == i_max:
            # last rule is removed
            explainer_copy.reduce_to_linear()
            tra_loss, _, _ = calculate_error(sam_train, sam_labels_train, explainer_copy, weight)
        else:
            # remove the rule with less significance
            explainer_copy.remove_rule()
            rem_rule_output, rem_fir_str_bar = rule_output[:, -(i + 1), :], fir_str_bar[:, -(i + 1)]
            tra_loss = fast_update_error(rem_rule_output, rem_fir_str_bar, sam_labels_train, prediction, weight)
        MR_his.append(copy.deepcopy(tra_loss))
        avg_tra_loss = avg_win(MR_his, 10)
        if avg_tra_loss > tra_loss0 * tau_R:
            break
        else:
            explainer = copy.deepcopy(explainer_copy)
            i += 1
    if explainer.is_linear:
        print('all rules are removed')
    if log is True:
        return explainer, i
    else:
        return explainer


def premise_condensation(explainer, membership_value, sam_train, sam_labels_train, tra_loss, tra_loss0, tau_P,
                            log=False, weight=None):
    # simplify the rules by removing redundant premises, please see Algorithm 2 in paper for more details
    MF_rem = np.mean(membership_value, axis=0)[0:-1, :]
    avg_tra_loss, MR_his, i = tra_loss, [tra_loss], 0
    explainer_copy = copy.deepcopy(explainer)
    i_max = explainer.num_fea * explainer.num_fuzzy_set - 1
    while i <= i_max:
        if i == i_max:
            # last premise is removed
            explainer_copy.reduce_to_linear()
        else:
            # remove the premise with less significance
            explainer_copy.remove_premise(MF_rem)
        tra_loss, _, _ = calculate_error(sam_train, sam_labels_train, explainer_copy, weight)
        MR_his.append(tra_loss)
        avg_tra_loss = avg_win(MR_his, 10)
        if avg_tra_loss > tra_loss0 * tau_P:
            break
        else:
            explainer = copy.deepcopy(explainer_copy)
            i += 1
    if explainer.is_linear:
        print('all premises are removed')
    if log is True:
        return explainer, i
    else:
        return explainer


def avg_win(MR_his, win_size):
    # calculate the average error of the last 'win_size' errors
    if len(MR_his) < win_size:
        avg_tra_acc = sum(MR_his) / len(MR_his)
    else:
        avg_tra_acc = sum(MR_his[-win_size:]) / win_size
    return avg_tra_acc


def load_dataset(name, mode='regression', num_features=5):
    # data loading and feature selection
    if mode == 'classification':
        dataset = np.loadtxt('cla_data/{}.csv'.format(name), delimiter=",", skiprows=0)
    else:
        dataset = np.loadtxt('reg_data/{}.csv'.format(name), delimiter=",", skiprows=0)
    sam, label = dataset[:, :-1], dataset[:, -1].reshape(-1, 1)
    if sam.shape[1] > num_features:
        fea_idx = forward_selection(sam, label, weights=None, num_features=5, random_state=1)
        sam = sam[:, fea_idx]
    train, test, labels_train, _ = \
        model_selection.train_test_split(sam, label, train_size=0.80, random_state=1)
    return train, test, labels_train


def forward_selection(data, labels, weights, num_features, random_state):
    # iteratively adds features to the model to select the most important features
    # this function derive from LIME [3]
    clf = Ridge(alpha=0, fit_intercept=True, random_state=random_state)
    used_features = []
    for _ in range(min(num_features, data.shape[1])):
        max_ = -100000000
        best = 0
        for feature in range(data.shape[1]):
            if feature in used_features:
                continue
            clf.fit(data[:, used_features + [feature]], labels,
                    sample_weight=weights)
            score = clf.score(data[:, used_features + [feature]],
                              labels,
                              sample_weight=weights)
            if score > max_:
                best = feature
                max_ = score
        used_features.append(best)
    return np.array(used_features)
