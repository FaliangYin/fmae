import pickle
import numpy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from fmae.utilities import load_dataset
import copy
from sklearn.cluster import KMeans
from fmae.fmae_upscaling import FmaeUpscaling
from references.evaluator import Faithfulness, Monotonicity, Infidelity

"""
The implementation of Section V.D Case 3 Upscaling via aggregation of the paper
"Hierarchical Fuzzy Model-Agnostic Explanation: Framework, Algorithms and Interface for XAI"
https://ieeexplore.ieee.org/document/10731553
"""

repeat_times = 10  # number of repeated experiments
mode = 'classification'  # use regression or classification datasets for experiments
if mode == 'regression':
    c = -1
    dataset_list = ['carbonnanotubes', 'HO', 'MO33test','SouthGermanCredit', 'WineQua_r',]
    score_name = 'R2_score'
else:
    c = 0
    dataset_list = ['YEAST', 'WBC', 'VOWEL', 'PIMA', 'PHONEME',]
    score_name = 'Accuracy'
record_list = ['training', 'testing(aggregation)', 'testing(direct)']
metrics_name = ['Faithfulness', 'Monotonicity', 'Infidelity']


def initial_records(metrics_name, score_name):
    """
    result_his: record the results of each batch
    result_avg: final result, the average value of all batches
    """
    temp_his, temp_avg = {}, {}
    for metric_name in metrics_name:
        temp_his[metric_name] = {'training': [], 'testing(aggregation)': [], 'testing(direct)': []}
        temp_avg[metric_name] = {'training': [], 'testing(aggregation)': [], 'testing(direct)': []}
    temp_his[score_name] = {'training': [], 'testing(aggregation)': [], 'testing(direct)': []}
    temp_avg[score_name] = {'training': [], 'testing(aggregation)': [], 'testing(direct)': []}
    return temp_his, temp_avg


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
    # identify the clusters to choose the instances
    kmeans = KMeans(n_clusters=repeat_times*2, n_init="auto")
    kmeans.fit(test)

    if mode == 'classification':
        model = MLPClassifier()  # store 'result_his' for all dataset
    else:
        model = MLPRegressor()  # closed box (black box) regressor
    model.fit(train, labels_train.ravel())

    scaler = StandardScaler(with_mean=False)
    scaler.fit(test)  # learn the distribution of the testing set for evaluation metrics requiring sampling

    results_his, results_avg = initial_records(metrics_name, score_name)
    for i in range(repeat_times):
        test_ = test[kmeans.labels_ == i]
        test_idx = numpy.random.choice(test_.shape[0], 20)  # choose instances from the cluster

        print('{} explained by {} in the {} of {} epoch'.format(dataset_name, 'FmaeUpscaling', i + 1, repeat_times))
        explainer = FmaeUpscaling
        if mode == 'classification':
            explainer = explainer(model.predict_proba, numpy.concatenate((train, test), axis=0), mode='classification')
        else:
            explainer = explainer(model.predict, numpy.concatenate((train, test), axis=0))

        # get a copy of the explainer to compare the results between
        # the explainer from aggregation and the explainer generated directly
        explainer_direct = copy.deepcopy(explainer)
        # the following explanations are corresponding to the results in Table V and VI in the paper
        # for domain, universe(direct) and universe(aggregation), respectively
        feature_weights_domain = explainer.explain(test_[test_idx[:10]])
        feature_weights_direct = explainer_direct.explain(test_[test_idx[-10:]])
        feature_weights_aggregation, score_w = explainer.weight_aggregation(test_[test_idx[-10:]])

        for metric_name in metrics_name:  # evaluate the generated explanations
            metric = eval(metric_name)
            if np.ndim(feature_weights_domain) == 3:  # if the feature_weights is [num_ioi, num_fea, num_class] or [num_ioi, num_fea]
                vals_domain, vals_aggregation, vals_direct = [], [], []
                for j in range(feature_weights_domain.shape[2]):
                    vals_domain.append(metric(model, test_[test_idx[:10]], feature_weights_domain[:, :, j], scaler=scaler, cla_idx=j))
                    vals_aggregation.append(metric(model, test_[test_idx[-10:]], feature_weights_aggregation[:, :, j], scaler=scaler, cla_idx=j))
                    vals_direct.append(metric(model, test_[test_idx[-10:]], feature_weights_direct[:, :, j], scaler=scaler, cla_idx=j))
                val_domain, val_aggregation, val_direct = np.average(vals_domain), np.average(vals_aggregation), np.average(vals_direct)
            else:
                val_domain = metric(model, test_[test_idx[:10]], feature_weights_domain, scaler=scaler)
                val_aggregation = metric(model, test_[test_idx[-10:]], feature_weights_aggregation, scaler=scaler)
                val_direct = metric(model, test_[test_idx[-10:]], feature_weights_direct, scaler=scaler)

            results_his[metric_name]['training'].append(val_domain)
            results_his[metric_name]['testing(aggregation)'].append(val_aggregation)
            results_his[metric_name]['testing(direct)'].append(val_direct)
        results_his[score_name]['training'].append(explainer.score)
        results_his[score_name]['testing(aggregation)'].append(score_w)
        results_his[score_name]['testing(direct)'].append(explainer_direct.score)

    for metric_name in metrics_name:
        for r in record_list:
            results_avg[metric_name][r] = [
                numpy.average(results_his[metric_name][r]),
                numpy.std(results_his[metric_name][r])]

    for r in record_list:
        results_avg[score_name][r] = [
                numpy.average(results_his[score_name][r]),
                numpy.std(results_his[score_name][r])]
    all_results_avg[dataset_name] = copy.deepcopy(results_avg)
    all_results_his[dataset_name] = copy.deepcopy(results_his)
    pass

# archive the experiment results
with open('results/'+mode+'/upscaling/avg_result.txt', 'w') as f:
    for dataset_name in dataset_list:
        f.write('Dataset name: {}\n'.format(dataset_name))
        for key, value in all_results_avg[dataset_name].items():
            f.write('{key}:{value}\n'.format(key=key, value=value))
        f.write('\n')

with open('results/'+mode+'/upscaling/all_results_avg.pkl', 'wb') as f:
    pickle.dump(all_results_avg, f)
with open('results/'+mode+'/upscaling/all_results_his.pkl', 'wb') as f:
    pickle.dump(all_results_his, f)
pass
