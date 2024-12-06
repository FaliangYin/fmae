import numpy as np
from sklearn.utils import check_random_state


def generate_samples(mask, num_samples, data_row, scaler, sample_around_instance=True, zoomer=1, standardization=False):
    # sample_around_instance False much worse
    if mask is None:
        mask = np.zeros_like(data_row)
    fid = np.where(mask)
    vid = np.where(mask == 0)
    num_cols = len(vid) # data_row.shape[0]

    instance_sample = data_row[list(vid[0])]
    scale = scaler.scale_[list(vid[0])]
    mean = scaler.mean_[list(vid[0])]

    random_state = check_random_state(None)
    data_rand = random_state.normal(
        0, 1, num_samples * num_cols).reshape(
        num_samples, num_cols)

    data = np.zeros((num_samples, len(mask)))
    if sample_around_instance:
        data[:, list(vid[0])] = data_rand * scale * zoomer + instance_sample
    else:
        data[:, list(vid[0])] = data_rand * scale * zoomer + mean
    data[0] = data_row.copy()
    data[:, list(fid[0])] = data_row[fid]

    if not standardization:
        return data
    else:
        return (data - scaler.mean_) / scaler.scale_


def predict_class(x, model, cla_idx):
    output = model.predict_proba(x)
    return output[:, cla_idx]


def Faithfulness(model, X, feature_weights, n_sample=100, ver='dec', con='observational', scaler=None, zoomer=1, cla_idx=-1):
    """
    Implementation of faithfulness, proposed by https://arxiv.org/abs/1806.07538
    Implementation from AIX360: https://github.com/Trusted-AI/AIX360
    For each datapoint, compute the correlation between the weights of the
    feature attribution algorithm, and the effect of the features on the
    performance of the model.
    TODO: add conditional expectation reference values
    """

    # X = X.numpy()
    if cla_idx == -1:
        f = model.predict
    else:
        f = lambda x: predict_class(x, model, cla_idx)

    num_datapoints, num_features = X.shape
    absolute_weights = abs(np.array(feature_weights))

    # compute the base values of each feature
    avg_feature_values = X.mean(axis=0)

    faithfulnesses = []
    for i in range(num_datapoints):
        """
        for each datapoint i, compute the correlation between feature weights
        and the delta in prediction when ablating each feature with replacement
        """
        # original prediction
        y_pred = np.squeeze(f(np.array([X[i]])))
        if ver == 'inc':
            y_pred = np.mean(np.squeeze(f(X)))

        # D new predictions (ablate one feature at a time)
        y_preds_new = np.zeros_like(X[i])

        for j in range(num_features):
            # generate a mask
            mask = np.ones_like(X[i])
            mask[j] = 0
            if ver == 'inc':
                mask = 1 - mask
            if con == "observational":
                # sample n_sample datapoints with feature j ablated
                x_sampled = generate_samples(mask, n_sample, X[i], scaler, zoomer=zoomer)
                # x_sampled = con_samples(mask, n_sample, X[i], X)
                # x_sampled, _ = generate_samples(mask, X[i], n_sample, X)
                # compute mean over n
                y_preds_new[j] = np.mean(np.squeeze(f(x_sampled)))
            elif con == "interventional":
                x_cond = avg_feature_values
                x_cond[mask.astype(bool)] = X[i][mask.astype(bool)]
                y_preds_new[j] = f([x_cond])[0]
        deltas = [abs(y_pred - y_preds_new[j]) for j in range(num_features)]
        faithfulness = np.corrcoef(absolute_weights[i], deltas)[0, 1]
        if np.isnan(faithfulness) or not np.isfinite(faithfulness):
            faithfulness = 0
        faithfulnesses.append(faithfulness)

    return np.mean(faithfulnesses)


def Monotonicity(model, X, feature_weights, n_sample=100, ver='dec', con='observational', scaler=None, zoomer=1, cla_idx=-1):
    """
    Implementation of monotonicity, proposed by https://arxiv.org/abs/1905.12698
    Implementation from AIX360: https://github.com/Trusted-AI/AIX360
    Check whether iteratively adding features from least weighted feature to most
    weighted feature, causes the prediction to monotonically improve.
    TODO: add other types of reference values

    Note: the default version measures the fraction of datapoints that *exactly*
    satisfy monotonicity. Setting avg=True is a bit more robust, since it measures
    how monotone each datapoint is. Both versions perform poorly on datasets where
    multiple features have roughly the same weights.
    """
    # X = X.numpy()
    if cla_idx == -1:
        f = model.predict
    else:
        f = lambda x: predict_class(x, model, cla_idx)

    num_datapoints, num_features = X.shape
    absolute_weights = abs(np.array(feature_weights))

    # compute the base values of each feature
    avg_feature_values = X.mean(axis=0)

    monotonicities = []
    y_preds_mean = np.mean(np.squeeze(f(X)))
    for i in range(num_datapoints):
        mask = np.zeros_like(X[i])
        sorted_weight_indices = np.argsort(absolute_weights[i])
        if ver == 'dec':
            sorted_weight_indices = sorted_weight_indices[::-1]
        y_preds_new = np.zeros(len(X[i]) + 1)
        y_preds_new[0] = y_preds_mean

        for j in sorted_weight_indices:
            mask[j] = 1
            if con == "observational":
                x_sampled = generate_samples(mask, n_sample, X[i], scaler, zoomer=zoomer)
                # x_sampled = con_samples(mask, n_sample, X[i], X)
                # x_sampled, _ = generate_samples(mask, X[i], n_sample, X)
                y_preds_new[j + 1] = np.mean(np.squeeze(f(x_sampled)))
            elif con == "interventional":
                x_cond = avg_feature_values
                x_cond[mask.astype(bool)] = X[i][mask.astype(bool)]
                y_preds_new[j + 1] = f([x_cond])[0]

        deltas = np.abs(np.diff(y_preds_new))
        if ver == 'dec':
            deltas = deltas[::-1]
        monotonicity = sum(np.diff(deltas) >= 0) / (num_features - 1)
        monotonicities.append(monotonicity)
    return np.mean(monotonicities)


def Infidelity(model, X, feature_weights, scaler=None, zoomer=1, cla_idx=-1):
    """
    Implementation of https://arxiv.org/pdf/1901.09392.pdf,
    based on https://github.com/chihkuanyeh/saliency_evaluation/blob/master/infid_sen_utils.py
    """
    def set_zero_infid(array, size, point, pert):
        ind = np.random.choice(size, point, replace=False)
        randd = np.random.normal(size=point) * 0.2 + array[ind]
        randd = np.minimum(array[ind], randd)
        randd = np.maximum(array[ind] - 1, randd)
        array[ind] -= randd
        return np.concatenate([array, ind, randd])

    def get_exp(ind, exp):
        return (exp[ind.astype(int)])

    # X = X.numpy()
    if cla_idx == -1:
        f = model.predict
    else:
        f = lambda x: predict_class(x, model, cla_idx)

    num_datapoints, num_features = X.shape
    infids = []

    for i in range(num_datapoints):
        num_reps = 1000
        x_orig = np.tile(X[i], [num_reps, 1])
        x = X[i]
        expl = feature_weights[i]
        expl_copy = np.copy(expl)
        val = np.apply_along_axis(set_zero_infid, 1, x_orig, num_features, num_features, pert="Gaussian")
        x_ptb, ind, rand = val[:, :num_features], val[:, num_features: 2 * num_features], val[:,
                                                                                          2 * num_features: 3 * num_features]
        exp_sum = np.sum(rand * np.apply_along_axis(get_exp, 1, ind, expl_copy), axis=1)
        ks = np.ones(num_reps)
        pdt = f([x])
        pdt_ptb = f(x_ptb)
        pdt_diff = pdt - pdt_ptb

        beta = np.mean(ks * pdt_diff * exp_sum) / np.mean(ks * exp_sum * exp_sum)
        exp_sum *= beta
        infid = np.mean(ks * np.square(pdt_diff - exp_sum)) / np.mean(ks)
        infids.append(infid)

    return np.mean(infids)
