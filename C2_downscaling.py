import pickle
import numpy
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from fmae.fmae_downscaling import FmaeDownscaling
from fmae.utilities import load_dataset
import copy
from references.evaluator import Faithfulness, Monotonicity, Infidelity

"""
The implementation of Section V.C Case 2 Downscaling via simplification of the paper
"Hierarchical Fuzzy Model-Agnostic Explanation: Framework, Algorithms and Interface for XAI"
https://ieeexplore.ieee.org/document/10731553
Authors: Faliang Yin, Hak-Keung Lam, David Watson
"""

num_ioi = 10   # number of instance of interest
repeat_times = 10  # number of repeated experiments
test_idx = numpy.random.randint(0, 100, [repeat_times, num_ioi])  # the index to select instances from testing set
mode = 'regression'  # use regression or classification datasets for experiments
if mode == 'regression':
    c = -1
    dataset_list = ['carbonnanotubes', 'HO', 'MO33test','SouthGermanCredit', 'WineQua_r',]
    score_name = 'R2_score'
else:
    c = 0
    dataset_list = ['YEAST', 'WBC', 'VOWEL', 'PIMA', 'PHONEME',]
    score_name = 'Accuracy'
metrics_name = ['Faithfulness', 'Monotonicity', 'Infidelity']
steps_name = ['initial', 'domain', 'local']


def initial_records(steps_name, metrics_name):
    """
    result_his: record the results of each batch
    result_avg: final result, the average value of all batches
    """
    results_his, results_avg = {}, {}
    for step_name in steps_name:
        temp_his, temp_avg = {}, {}
        for metric_name in metrics_name:
            temp_his[metric_name] = []
            temp_avg[metric_name] = []
        temp_his['score'], temp_avg['score'] = [], []  # R2_score for regression and Accuracy for classification
        temp_his['rule_num'], temp_avg['rule_num'] = [], []  # number of rules
        temp_his['rem_pre'], temp_avg['rem_pre'] = [], []  # number of the removed premises
        results_his[step_name], results_avg[step_name] = temp_his, temp_avg
    return results_his, results_avg


all_results_avg, all_results_his = {}, {}
for dataset_name in dataset_list:
    all_results_avg[dataset_name] = []  # store 'result_avg' for all dataset
    all_results_his[dataset_name] = []  # store 'result_his' for all dataset
all_log_his = {}

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

    results_his, results_avg = initial_records(steps_name, metrics_name)
    log_his = []
    for i in range(repeat_times):
        print('{} explained by {} in the {} of {} epoch'.format(dataset_name, 'FMAE_downscaling', i + 1, repeat_times))
        if mode == 'classification':
            explainer = FmaeDownscaling(model.predict_proba, train, mid_flag=True, mode=mode)
        else:
            explainer = FmaeDownscaling(model.predict, train, mid_flag=True)
        initial_, domain_, local_, num_rule_all, _ = explainer.explain(test[test_idx[i]])  # generate explanations
        log_his.append(num_rule_all)
        for metric_name in metrics_name:  # evaluate the generated explanations
            metric = eval(metric_name)
            if np.ndim(initial_[0]) == 3:  # if the feature_weights is [num_ioi, num_fea, num_class] or [num_ioi, num_fea]
                val0s, val1s, val_s = [], [], []
                for j in range(initial_[0].shape[2]):
                    val0s.append(metric(model, test[test_idx[i]], initial_[0][:, :, j], scaler=scaler, cla_idx=j))  # , zoomer=explainer.zoomer
                    val1s.append(metric(model, test[test_idx[i]], local_[0][:, :, j], scaler=scaler, zoomer=explainer.zoomer, cla_idx=j))
                    val_s.append(metric(model, test[test_idx[i]], domain_[0][:, :, j], scaler=scaler, cla_idx=j))
                val0, val1, val_ = np.average(val0s), np.average(val1s), np.average(val_s)
            else:
                val0 = metric(model, test[test_idx[i]], initial_[0], scaler=scaler, cla_idx=c)
                val1 = metric(model, test[test_idx[i]], local_[0], scaler=scaler, zoomer=explainer.zoomer, cla_idx=c)
                val_ = metric(model, test[test_idx[i]], domain_[0], scaler=scaler, cla_idx=c)
            results_his['initial'][metric_name].append(val0)
            results_his['local'][metric_name].append(val1)
            results_his['domain'][metric_name].append(val_)

        results_his['initial']['score'].append(np.average(initial_[1]))
        results_his['local']['score'].append(np.average(local_[1]))
        results_his['domain']['score'].append(np.average(domain_[1]))
        results_his['initial']['rule_num'].append(np.average(initial_[2]))
        results_his['local']['rule_num'].append(np.average(local_[2]))
        results_his['domain']['rule_num'].append(np.average(domain_[2]))
        results_his['initial']['rem_pre'].append(0)
        results_his['local']['rem_pre'].append(np.average(local_[3]))
        results_his['domain']['rem_pre'].append(np.average(domain_[3]))

    all_log_his[dataset_name] = log_his

    for step_name in steps_name:  # calculate the average performance for the repeated experiments
        for metric_name in metrics_name:
            results_avg[step_name][metric_name] = [numpy.average(results_his[step_name][metric_name]),
                                                   numpy.std(results_his[step_name][metric_name])]
        results_avg[step_name]['score'] = [numpy.average(results_his[step_name]['score']),
                                              numpy.std(results_his[step_name]['score'])]

        results_avg[step_name]['rule_num'] = [numpy.average(results_his[step_name]['rule_num']),
                                              numpy.std(results_his[step_name]['rule_num'])]

        results_avg[step_name]['rem_pre'] = [numpy.average(results_his[step_name]['rem_pre']),
                                             numpy.std(results_his[step_name]['rem_pre'])]
    all_results_avg[dataset_name] = copy.deepcopy(results_avg)
    all_results_his[dataset_name] = copy.deepcopy(results_his)
    pass

# archive the experiment results
with open('results/'+mode+'/downscaling/avg_result.txt', 'w') as f:
    for dataset_name in dataset_list:
        f.write('Dataset name: {}\n'.format(dataset_name))
        for key, value in all_results_avg[dataset_name].items():
            f.write('{key}:{value}\n'.format(key=key, value=value))
        f.write('\n')

with open('results/'+mode+'/downscaling/all_results_his.pkl', 'wb') as f:
    pickle.dump(all_results_his, f)
with open('results/'+mode+'/downscaling/all_log_his.pkl', 'wb') as f:
    pickle.dump(all_log_his, f)
pass
