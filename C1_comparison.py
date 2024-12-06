import pickle
import numpy
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from fmae.utilities import load_dataset
import copy

from references.ground_truth_shap import BruteForceKernelShap
from references.lime import Lime
from references.maple_c import Maple_c
from references.maple_r import Maple_r
from references.shap import Shap, KernelShap
from fmae.fmae_tabular import FmaeFls, FmaeExplainer
from references.evaluator import Faithfulness, Monotonicity, Infidelity

"""
The implementation of Section V.B Case 1 Comparisons with mainstream methods of the paper
"Hierarchical Fuzzy Model-Agnostic Explanation: Framework, Algorithms and Interface for XAI"
https://ieeexplore.ieee.org/document/10731553
Authors: Faliang Yin, Hak-Keung Lam, David Watson
"""

repeat_times = 10  # number of repeated experiments (number of batch)
num_ioi = 10   # number of instance of interest (size of batch)
test_idx = numpy.random.randint(0, 100, [repeat_times, num_ioi])  # the index to select instances from testing set
mode = 'regression'  # use regression or classification datasets for experiments
if mode == 'regression':
    c = -1
    Maple_type = 'Maple_r'
    dataset_list = ['carbonnanotubes', 'HO', 'MO33test','SouthGermanCredit', 'WineQua_r',]
    score_name = 'R2_score'
else:
    c = 0
    Maple_type = 'Maple_c'
    dataset_list = ['YEAST', 'WBC', 'VOWEL', 'PIMA', 'PHONEME',]
    score_name = 'Accuracy'
explainer_names = ['BruteForceKernelShap', 'KernelShap', 'Shap', Maple_type, 'Lime', 'FmaeFls', 'FmaeExplainer', ]
metric_names = ['Faithfulness', 'Monotonicity', 'Infidelity']


def initial_records(explainers_name, metrics_name, score_name):
    """
    result_his: record the results of each batch
    result_avg: final result, the average value of all batches
    """
    results_his, results_avg = {}, {}
    for explainer_name in explainers_name:
        temp_his, temp_avg = {}, {}
        for metric_name in metrics_name:
            temp_his[metric_name] = []
            temp_avg[metric_name] = []
        temp_his[score_name], temp_avg[score_name] = [], []
        results_his[explainer_name], results_avg[explainer_name] = temp_his, temp_avg
    return results_his, results_avg


all_results_avg, all_results_his, num_rule_his = {}, {}, {}
for dataset_name in dataset_list:
    all_results_avg[dataset_name] = []  # store 'result_avg' for all dataset
    all_results_his[dataset_name] = []  # store 'result_his' for all dataset
    num_rule_his[dataset_name] = []  # record the number of rules for FMAE

for dataset_name in dataset_list:
    print('-----------------------------')
    print('Start the explanation of dataset {}'.format(dataset_name))
    print('-----------------------------')

    train, test, labels_train = load_dataset(dataset_name, mode)
    if mode == 'classification':
        model = MLPClassifier()  # closed box (black box) classifier
    else:
        model = MLPRegressor()  # closed box (black box) regressor
    model.fit(train, labels_train.ravel())

    scaler = StandardScaler(with_mean=False)
    scaler.fit(test)  # learn the distribution of the testing set for evaluation metrics requiring sampling

    results_his, results_avg = initial_records(explainer_names, metric_names, score_name)

    for i in range(repeat_times):
        for explainer_name in explainer_names:
            print('{} explained by {} in the {} of {} epoch'.format(dataset_name, explainer_name, i + 1, repeat_times))
            explainer = eval(explainer_name)
            if mode == 'classification':
                if explainer_name in ['Lime', 'FmaeFls', 'FmaeExplainer']:
                    explainer = explainer(model.predict_proba, numpy.concatenate((train, test), axis=0), mode='classification')
                else:
                    explainer = explainer(model.predict_proba, numpy.concatenate((train, test), axis=0))
            else:
                explainer = explainer(model.predict, numpy.concatenate((train, test), axis=0))
            feature_weights = explainer.explain(test[test_idx[i]])  # generate explanations

            for metric_name in metric_names:  # evaluate the generated explanations
                metric = eval(metric_name)
                if np.ndim(feature_weights) == 3:  # if the feature_weights is [num_ioi, num_fea, num_class] or [num_ioi, num_fea]
                    vals = []
                    for j in range(feature_weights.shape[2]):
                        vals.append(metric(model, test[test_idx[i]], feature_weights[:, :, j], scaler=scaler, cla_idx=j))
                    val = np.average(vals)
                else:
                    val = metric(model, test[test_idx[i]], feature_weights, scaler=scaler, cla_idx=c)
                results_his[explainer_name][metric_name].append(val)
            if explainer_name in ['Lime', 'FmaeFls', 'FmaeExplainer']:
                results_his[explainer_name][score_name].append(explainer.score)
            else:
                results_his[explainer_name][score_name].append(0)
            if explainer_name == 'FmaeExplainer':
                num_rule_his[dataset_name].append(explainer.num_rule)

    for explainer_name in explainer_names:  # calculate the average performance for the repeated experiments
        for metric_name in metric_names:
            results_avg[explainer_name][metric_name] = [numpy.average(results_his[explainer_name][metric_name]),
                                                        numpy.std(results_his[explainer_name][metric_name])]
        results_avg[explainer_name][score_name] = [numpy.average(results_his[explainer_name][score_name]),
                                                   numpy.std(results_his[explainer_name][score_name])]
    num_rule_his[dataset_name].append(np.average(num_rule_his[dataset_name]))
    all_results_avg[dataset_name] = copy.deepcopy(results_avg)
    all_results_his[dataset_name] = copy.deepcopy(results_his)
    pass

# archive the experiment results
with open('results/'+mode+'/comparison/all_results_his.pkl', 'wb') as f:
    pickle.dump(all_results_his, f)
with open('results/'+mode+'/comparison/all_results_avg.pkl', 'wb') as f:
    pickle.dump(all_results_avg, f)
with open('results/'+mode+'/comparison/num_rule_his.pkl', 'wb') as f:
    pickle.dump(num_rule_his, f)

with open('results/'+mode+'/comparison/avg_result.txt', 'w') as f:
    for dataset_name in dataset_list:
        f.write('Dataset name: {}\n'.format(dataset_name))
        for key, value in all_results_avg[dataset_name].items():
            f.write('{key}:{value}\n'.format(key=key, value=value))
        f.write('\n')
pass
