from sklearn.metrics import r2_score, accuracy_score
from fmae.utilities import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

"""
FMAE for tabular explanation task with SINGLE instance (tabular data)
FmaeFls: corresponding to 'Initial FLS' in Fig.2 in paper
    where rule reduction and premise condensation are not enabled
FmaeExplainer: corresponding to 'Initial explainer' in Fig.2 in paper
    where rule reduction and premise condensation can be enabled

Please refer to the paper for more details:
"Hierarchical Fuzzy Model-Agnostic Explanation: Framework, Algorithms and Interface for XAI"
https://ieeexplore.ieee.org/document/10731553
Authors: Faliang Yin, Hak-Keung Lam, David Watson
"""


def membership_fun(x, m):
    # Gaussian membership function
    return np.exp((-(x - m) ** 2))


class FmaeBasicFls:
    """
    The basic TSK FLS model of FMAE
    Parameters
    ----------
    num_fea: number of features
    num_class: number of classes
    num_fuzzy_set: number of fuzzy sets
    mode: regression or classification
    """
    def __init__(self, num_fea, num_class, num_fuzzy_set, mode='regression'):

        super().__init__()
        self.mode = mode  # regression or classification
        self.num_fea = num_fea  # number of input features
        self.num_class = num_class  # number of output (classes)
        self.num_fuzzy_set = num_fuzzy_set  # number of fuzzy set for each feature

        # generate the rule base
        rule_base = np.zeros([num_fuzzy_set ** num_fea, num_fea])  # fuzzy rule base
        for i, ii in enumerate(reversed(range(num_fea))):  # i--Ascending subscript, ii--descending subscript
            rule_base[:, ii] = np.tile(np.repeat(range(num_fuzzy_set), num_fuzzy_set ** i), num_fuzzy_set ** ii)
        self.rule_base = rule_base.astype(np.int32)
        self.num_rule = self.rule_base.shape[0]  # number of rules

        # initialize the parameters
        self.center = np.random.rand(self.num_fuzzy_set, self.num_fea)  # center of fuzzy sets
        self.center_mode = 'equidistant'  # equidistant or k-means
        self.con_param = np.zeros([self.num_class, self.num_rule, self.num_fea + 1])  # consequent parameters

        self.coef_ = np.zeros(num_fea)  # feature salience values
        self.intercept_ = 0.  # intercept in feature salience explanation

        self.rule_reduction = False  # if the rule_reduction is allowed
        self.premise_condensation = False  # if the premise_condensation is allowed
        self.is_linear = False  # if the explainer is reduced to a linear system

        self.scaler = StandardScaler()  # scaler to standardize the data

    def ini_center(self, original_input):
        # initialize the centers of fuzzy sets
        scaled_input = self.scaler.fit_transform(original_input)
        if self.center_mode == 'k-means':
            # set the centers by k-means
            kmeans = KMeans(n_clusters=self.num_fuzzy_set, n_init="auto")
            kmeans.fit(scaled_input)
            self.center = kmeans.cluster_centers_
        else:
            # take the centers equidistant from the minimum to maximum value of the samples
            min_fea = np.min(scaled_input, axis=0)
            max_fea = np.max(scaled_input, axis=0)
            for i in range(self.num_fea):
                self.center[:, i] = np.linspace(min_fea[i], max_fea[i], self.num_fuzzy_set)

    def antecedent(self, original_input):
        # antecedent to calculate the firing strengths
        scaled_input = self.scaler.transform(original_input)
        num_sam = scaled_input.shape[0]
        membership_value = membership_fun(np.expand_dims(scaled_input, axis=1),
                                          self.center)
        membership_value = np.concatenate([membership_value, np.ones([num_sam, 1, self.num_fea])], axis=1)

        in_dim, fs_ind = self.num_fea, self.rule_base
        fir_str = np.prod(membership_value[:, fs_ind, range(in_dim)], axis=2)
        fir_str_bar = fir_str / np.expand_dims(np.sum(fir_str, 1), axis=1)
        return fir_str_bar, membership_value

    def consequent(self, original_input):
        # consequent containing a set of linear system
        scaled_input = self.scaler.transform(original_input)
        rule_output = (self.con_param[:, :, 1:] @ scaled_input.T
                       ).T + self.con_param[:, :, 0].T
        return rule_output

    def forward(self, original_input):
        # forward propagation of the FLS
        if not self.is_linear:
            fir_str_bar, membership_value = self.antecedent(original_input)
            rule_output = self.consequent(original_input)
            model_output = np.einsum('NRC,NR->NC', rule_output, fir_str_bar)
        else:
            fir_str_bar, membership_value = 1, 1
            model_output = np.squeeze(self.consequent(original_input), axis=1)
        return model_output, fir_str_bar, membership_value

    def predict(self, original_input):
        # use the FLS to predict
        model_output, _, _ = self.forward(original_input)
        return model_output

    def score(self, samples, labels, sample_weight=None):
        # evaluate approximation ability of the FLS
        prediction = self.predict(samples)
        if self.mode == 'regression':
            r2 = r2_score(labels, prediction, sample_weight=sample_weight)
            # rmse = mean_squared_error(labels, prediction, sample_weight=sample_weight)
            r = []
            for i in range(self.num_class):
                r.append(r2_score(labels[:, i], prediction[:, i], sample_weight=sample_weight))
            return r2
        else:
            acc = accuracy_score(labels.argmax(axis=1), prediction.argmax(axis=1), sample_weight=sample_weight)
            return acc

    def remove_rule(self):
        # remove the last rule for rule reduction; see Algorithm 2 in paper
        self.con_param = self.con_param[:, :-1, :]
        self.rule_base = self.rule_base[:-1, :]
        self.num_rule = self.rule_base.shape[0]

    def remove_premise(self, membership_value):
        # find the removing index for premise condensation; see Algorithm 2 in paper
        max_pre = np.max(membership_value, axis=1)
        rem_dim = np.argmax(membership_value, axis=1)
        rem_fs = np.argmax(max_pre, axis=0)
        rem_dim = rem_dim[rem_fs]
        membership_value[rem_fs, rem_dim] = 0

        rem_dim_rule_base = self.rule_base[:, rem_dim]
        rem_dim_rule_base[rem_dim_rule_base == rem_fs] = self.num_fuzzy_set
        self.rule_base[:, rem_dim] = rem_dim_rule_base
        return [rem_dim, rem_fs]

    def reduce_to_linear(self):
        # reduce the FLS to a linear model
        self.con_param = np.sum(self.con_param, axis=1, keepdims=True)
        self.is_linear = True
        self.num_rule = 0
        self.rule_base = []

    def lse_con_param(self, original_input, target_output, sample_weight=None, lbd=0.1):
        # train the consequent parameters by least squares estimation
        scaled_input = self.scaler.transform(original_input)
        model_input_plus = np.concatenate([np.ones([scaled_input.shape[0], 1]), scaled_input],
                                          axis=1)
        if self.is_linear:
            if sample_weight is None:
                con_param_temp0 = np.linalg.inv(model_input_plus.T @ model_input_plus +
                                               lbd * np.eye(model_input_plus.shape[1])) \
                                 @ model_input_plus.T
                for i in range(self.num_class):
                    self.con_param[i, :, :] = (con_param_temp0 @ target_output[:, i]).reshape(1, self.num_fea + 1)
            else:
                # self.sample_weight = sample_weight
                sample_weight = np.diag(sample_weight)
                con_param_temp0 = np.linalg.inv(model_input_plus.T @ sample_weight @ model_input_plus +
                                                lbd * np.eye(model_input_plus.shape[1])) \
                                  @ model_input_plus.T @ sample_weight
                for i in range(self.num_class):
                    self.con_param[i, :, :] = (con_param_temp0 @ target_output[:, i]).reshape(1, self.num_fea + 1)
            return
        fir_str_bar, _ = self.antecedent(original_input)
        fir_str_bar_input = np.repeat(fir_str_bar, repeats=model_input_plus.shape[1],
                                      axis=1) * np.tile(model_input_plus, [1, fir_str_bar.shape[1]])

        if sample_weight is None:
            con_param_temp = np.linalg.inv(fir_str_bar_input.T @ fir_str_bar_input +
                                           lbd * np.eye(fir_str_bar_input.shape[1])) @ fir_str_bar_input.T \
                             @ target_output
            self.con_param = np.expand_dims(con_param_temp, axis=0).reshape(self.con_param.shape)
        else:
            # self.sample_weight = sample_weight
            sample_weight = np.diag(sample_weight)
            con_param_temp0 = np.linalg.inv(fir_str_bar_input.T @ sample_weight @ fir_str_bar_input +
                                           lbd * np.eye(fir_str_bar_input.shape[1])) @ fir_str_bar_input.T \
                             @ sample_weight
            for i in range(self.num_class):
                self.con_param[i, :, :] = (con_param_temp0 @ target_output[:, i]).reshape(self.num_rule, self.num_fea + 1)

    def fit(self, original_input, target_output, sample_weight=None):
        # train the FLS
        self.ini_center(original_input)
        self.lse_con_param(original_input, target_output, sample_weight)
        explained_instance = np.expand_dims(original_input[0], axis=0)
        self.feature_attribution(explained_instance)

    def feature_attribution(self, original_input, rule_mode=False):
        # generate feature salience explanations
        if self.is_linear:
            fs = np.squeeze(self.con_param[:, :, 1:]).T
            self.coef_ = fs
            self.intercept_ = np.squeeze(self.con_param[:, :, 0])
            return self.coef_
        fir_str_bar, _ = self.antecedent(original_input)
        fs = np.einsum('CRD,NR->NDC', self.con_param[:, :, 1:], fir_str_bar)
        if rule_mode is True:
            fs = np.einsum('CRD,NR->NDCR', self.con_param[:, :, 1:], fir_str_bar)
        if original_input.shape[0] == 1:
            fs = np.squeeze(fs, axis=0)
        else:  # multiple input
            fs = np.mean(fs, axis=0)
        if fs.shape[1] == 1: # regression
            self.coef_ = np.squeeze(fs, axis=1)
            self.intercept_ = np.mean(np.squeeze((fir_str_bar @ self.con_param[:, :, 0].T), axis=1))
        else: # classification
            self.coef_ = fs
            self.intercept_ = np.mean((fir_str_bar @ self.con_param[:, :, 0].T), axis=0)
        return self.coef_


class FmaeBasicExplainer:
    """
    The basic explainer of FMAE
    Parameters
    ----------
    num_fea: number of features
    num_class: number of classes
    num_fuzzy_set: number of fuzzy sets
    flag: choose the algorithm for simplification
        'initial': do not simplify the fuzzy rules
        'rule_reduction': only conduct rule reduction
        'premise_condensation': conduct both rule reduction and premise condensation
    mode: regression or classification
    """
    def __init__(self, num_fea, num_class, num_fuzzy_set, flag='premise_condensation', mode='regression'):

        self.explainer = FmaeBasicFls(num_fea, num_class, num_fuzzy_set, mode)  # basic TSK FLS model of FMAE
        self.flag = flag  # choose the algorithm for simplification
        self.coef_ = None  # feature salience values
        self.intercept_ = None # intercept in feature salience explanation
        self.num_rule = 0  # number of rules
        self.rem_rule = 0  # number of the removed rules
        self.rem_pre = 0  # number of the removed premises

    def fit(self, samples, labels, sample_weight=None):
        # train the explainer
        self.explainer.ini_center(samples)
        self.explainer.lse_con_param(samples, labels, sample_weight)

        if self.flag == 'rule_reduction' or self.flag == 'premise_condensation':
            # rule reduction
            tra_loss_ini, fir_str_bar, membership_value = calculate_error(samples, labels, self.explainer,
                                                                          sample_weight)
            self.explainer, self.rem_rule = rule_reduction(self.explainer, fir_str_bar, samples, labels, tra_loss_ini,
                                                           tra_loss_ini, tau_R=1.07, log=True, weight=sample_weight)
            self.explainer, tra_loss, _ = tune_model(self.explainer, samples, labels, sample_weight=sample_weight)
            if self.flag == 'premise_condensation' and not self.explainer.is_linear:
                # premise condensation
                self.explainer, self.rem_pre = premise_condensation(self.explainer, membership_value, samples, labels,
                                                                   tra_loss, tra_loss_ini, tau_P=1.1, log=True,
                                                                   weight=sample_weight)
                self.explainer, _, _ = tune_model(self.explainer, samples, labels, sample_weight=sample_weight)

        explained_instance = np.expand_dims(samples[0], axis=0)
        self.explainer.feature_attribution(explained_instance)
        self.intercept_ = self.explainer.intercept_
        self.coef_ = self.explainer.coef_
        self.num_rule = self.explainer.num_rule

    def score(self, samples, labels, sample_weight=None):
        return self.explainer.score(samples, labels, sample_weight)

    def predict(self, model_input):
        return self.explainer.predict(model_input)

    def feature_attribution(self, model_input):
        return self.explainer.feature_attribution(model_input)
