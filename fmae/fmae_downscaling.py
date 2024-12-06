from fmae.fmae_base import FmaeBasicFls
from fmae.utilities import *
import sklearn

"""
FMAE for downscaling the explanation from domain level to local level
FmaeDownscaling: corresponding to the experiment of Section V.C Case 2 Downscaling via simplification of the paper

Please refer to the paper for more details:
"Hierarchical Fuzzy Model-Agnostic Explanation: Framework, Algorithms and Interface for XAI"
https://ieeexplore.ieee.org/document/10731553
Authors: Faliang Yin, Hak-Keung Lam, David Watson
"""


class FmaeDownscaling:
    """
    Providing the explanations of the hierarchical FMAE framework in a top-down manner (domain to local).
    Parameters
    ----------
    model: closed box (black box) model to be explained
    instances: instances of interest to be explained
    mode: regression or classification
    num_sam: number of samples
    num_fuzzy_set: number of fuzzy set for each feature
    mid_flag: whether generating domain explainer:
        if True, the initial FLS will first be simplified to a domain explainer,
        and then be downscaled to a local explainer;
        if False, the initial FLS will be simplified and downscaled to a local explainer.
    """
    def __init__(self, model, instances, mode='regression', num_sam=5000, num_fuzzy_set=3, mid_flag=True):

        self.model = model  # closed box (black box) model to be explained
        if str(type(instances)).endswith("pandas.core.frame.DataFrame'>"):  # transform to np data if input is pd data
            instances = instances.values
        self.mode = mode  # regression or classification
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(instances)
        self.zoomer = 1  # sampling scale corresponding to explanation scope when reduce the level from domain to local
        self.num_sam = num_sam  # number of samples
        self.kernel_width = 0.5 * np.sqrt(instances.shape[-1])  # kernel width for sampling

        self.num_fea = instances.shape[1]  # number of input features
        self.num_fuzzy_set = num_fuzzy_set  # number of fuzzy set for each feature
        self.num_class = 1  # number of output (classes)
        self.explainer = None  # basic TSK FLS model of FMAE
        self.score = None  # performance score for approximation ability of the FLS
        self.mid = mid_flag  # whether generating domain explainer

    def explain(self, instances):
        if str(type(instances)).endswith("pandas.core.frame.DataFrame'>"):
            instances = instances.values
        fs_all, num_rule_all = [], []  # record the change of feature salience values and rule number for all instance
        zoomer_all = []  # record the change of sampling scale
        # record feature salience values, approximate performance score, rule number
        # and number of removed premise in initial FLS, domain explainer and local explainer
        fs_initial, score_initial, rule_initial = [], [], 0
        fs_domain, score_domain, rule_domain, rem_pre_domain = [], [], [], []
        fs_local, score_local, rule_local, rem_pre_local = [], [], [], []
        for instance in instances:
            fs_record, num_rule_record = [], []  # record feature salience values and rule number in each step
            tau_r, zoomer, j = 1.07, 1, 0  # the threshold for rule reduction, sampling scale

            # train the initial FLS
            # for each instance, generate samples, labels and weights to train an FLS as explainer
            samples = get_samples(instance, self.num_sam, self.scaler, zoomer=zoomer)
            labels = self.model(samples)
            if self.mode == 'regression':
                labels = np.expand_dims(labels, axis=1)
            else:
                self.num_class = labels.shape[1]
            self.explainer = FmaeBasicFls(self.num_fea, self.num_class, self.num_fuzzy_set, self.mode)
            weights = get_weights(samples, self.scaler, kernel_width=self.kernel_width)
            self.explainer.fit(samples, labels, weights)
            fs = self.explainer.coef_  # generate feature salience explanations
            fs_record.append(fs)
            num_rule_record.append(self.explainer.num_rule)
            tra_loss, fir_str_bar, membership_value = calculate_error(samples, labels, self.explainer, weight=weights)
            tra_loss_initial = tra_loss.copy()
            fs_initial.append(fs)
            score_initial.append(self.explainer.score(samples, labels, sample_weight=weights))
            rule_initial = self.explainer.num_rule

            # simplify the initial FLS to domain explainer
            self.explainer, _ = rule_reduction(self.explainer, fir_str_bar, samples, labels, tra_loss, tra_loss_initial,
                                               tau_r, log=True, weight=weights)
            if self.mid:  # if a domain explainer is required
                explainer_domain = copy.deepcopy(self.explainer)
                explainer_domain.lse_con_param(samples, labels, weights)
                if not explainer_domain.is_linear:
                    tra_loss_domain, _, _ = calculate_error(samples, labels, explainer_domain, weight=weights)
                    explainer_domain, rem_num = premise_condensation(explainer_domain, membership_value, samples,
                                                                     labels, tra_loss_domain, tra_loss_initial,
                                                                     tau_P=1.1, log=True, weight=weights)
                    explainer_domain.lse_con_param(samples, labels, weights)
                else:
                    rem_num = 0
                fs = self.explainer.feature_attribution(np.expand_dims(samples[0], axis=0))  # generate feature salience explanations
                fs_domain.append(fs)
                score_domain.append(explainer_domain.score(samples, labels, sample_weight=weights))
                rule_domain.append(explainer_domain.num_rule)
                rem_pre_domain.append(rem_num)
                pass

            # downscale the explainer to local level
            while zoomer >= 0.5:  # minimum value for downscaling the explanation scope
                zoomer -= 0.1
                samples = get_samples(instance, self.num_sam, self.scaler, zoomer=zoomer)
                labels = self.model(samples)
                if self.mode == 'regression':
                    labels = np.expand_dims(labels, axis=1)
                weights = get_weights(samples, self.scaler, kernel_width=self.kernel_width)
                self.explainer.fit(samples, labels, weights)
                fs = self.explainer.coef_
                fs_record.append(fs)
                num_rule_record.append(self.explainer.num_rule)
                if self.explainer.is_linear:  # conduct rule reduction if FLS is not reduced to linear system
                    break
                tra_loss, fir_str_bar, membership_value = calculate_error(samples, labels, self.explainer,
                                                                          weight=weights)
                self.explainer, _ = rule_reduction(self.explainer, fir_str_bar, samples, labels, tra_loss,
                                                   tra_loss_initial, tau_r, log=True, weight=weights)
                j += 1

            if not self.explainer.is_linear:  # conduct premise condensation if FLS is not reduced to linear system
                self.explainer, rem_num = premise_condensation(self.explainer, membership_value, samples, labels,
                                                               tra_loss, tra_loss_initial, tau_P=1.1, log=True,
                                                               weight=weights)
                if rem_num != 0:  # if at least one premise is removed after premise condensation
                    self.explainer.lse_con_param(samples, labels, weights)
                    fs = self.explainer.feature_attribution(np.expand_dims(samples[0], axis=0))  # generate feature salience explanations
                    fs_record.append(fs)
                    num_rule_record.append(rem_num)
                else:
                    num_rule_record.append(0)
            else:
                num_rule_record.append(-1)
                rem_num = 0
            rule_local.append(self.explainer.num_rule)
            fs_local.append(fs)
            score_local.append(self.explainer.score(samples, labels, sample_weight=weights))
            rem_pre_local.append(rem_num)

            fs_all.append(fs_record)
            num_rule_all.append(num_rule_record)
            zoomer_all.append(zoomer)

        self.zoomer = np.mean(zoomer_all)  # calculate the average explanation scope for evaluation metrics later
        self.score = np.average(score_local)
        initial_ = [np.array(fs_initial), score_initial, rule_initial]
        domain_ = [np.array(fs_domain), score_domain, rule_domain, rem_pre_domain]
        local_ = [np.array(fs_local), score_local, rule_local, rem_pre_local]
        if self.mid:
            return initial_, domain_, local_, num_rule_all, fs_all
        else:
            return np.array(fs_local)
