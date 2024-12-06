from fmae.fmae_base import FmaeBasicFls
from fmae.utilities import *
import sklearn


class FMAE_downscaling:
    def __init__(self, model, samples, mid_info=False, mode='regression'):
        self.explainer = None
        self.model = model
        if str(type(samples)).endswith("pandas.core.frame.DataFrame'>"):
            samples = samples.values
        self.mode = mode
        self.num_fea = samples.shape[1]
        self.num_fuzzy_set = 3
        self.num_class = 1
        self.mid_info = mid_info

        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(samples)
        self.zoomer = 1
        self.num_sam = 5000
        self.kernel_width = 0.5 * np.sqrt(samples.shape[-1])
        self.score = None
        self.mid = True

    def explain(self, x):
        if str(type(x)).endswith("pandas.core.frame.DataFrame'>"):
            x = x.values
        fs_all, log_all = [], []
        fs_ini, score_ini = [], []
        fs_mid, score_mid, rule_mid, rem_pre_mid = [], [], [], []
        fs_end, score_end, rule_end, rem_pre_end = [], [], [], []
        zoomer_all = []
        for i in range(x.shape[0]):
            fs_hier, log_hier = [], []
            # rule reduction
            tau_M, zoomer, j = 1.07, 1.1, 0
            while zoomer >= 0.5:
                zoomer -= 0.1
                samples = get_samples(x[i], self.num_sam, self.scaler, zoomer=zoomer)
                labels = self.model(samples)
                if self.mode == 'regression':
                    labels = np.expand_dims(labels, axis=1)
                else:
                    self.num_class = labels.shape[1]
                if j == 0:
                    self.explainer = FmaeBasicFls(self.num_fea, self.num_class, self.num_fuzzy_set, self.mode)
                weights = get_weights(samples, self.scaler, kernel_width=self.kernel_width)

                # train model
                self.explainer.ini_center(samples)
                self.explainer.lse_con_param(samples, labels, weights)

                if self.mode == 'regression':
                    fs = self.explainer.feature_attribution(np.expand_dims(x[i], axis=0))  #
                else:
                    fs = self.explainer.feature_attribution(np.expand_dims(x[i], axis=0))
                fs_hier.append(fs)
                log_hier.append(self.explainer.num_rule)

                if self.explainer.is_linear:
                    break

                tra_loss, fir_str_bar, membership_value = calculate_error(samples, labels, self.explainer,
                                                                          weight=weights)
                if j == 0:
                    tra_loss_ini = tra_loss.copy()
                    fs_ini.append(fs)
                    score_ini.append(self.explainer.score(samples, labels, sample_weight=weights))
                    rule_ini = self.explainer.num_rule

                self.explainer, rem_num = rule_reduction(self.explainer, fir_str_bar, samples, labels, tra_loss,
                                                         tra_loss_ini, tau_M, log=True, weight=weights)

                if j == 0 and self.mid:
                    explainer2 = copy.deepcopy(self.explainer)
                    explainer2.lse_con_param(samples, labels, weights)
                    if not explainer2.is_linear:
                        tra_loss2, _, _ = calculate_error(samples, labels, explainer2, weight=weights)
                        explainer2, rem_num2 = premise_condensation(explainer2, membership_value, samples, labels,
                                                                    tra_loss2, tra_loss_ini, tau_R=1.1, log=True,
                                                                    weight=weights)
                        explainer2.lse_con_param(samples, labels, weights)
                    else:
                        rem_num2 = 0
                    if self.mode == 'regression':
                        fs2 = explainer2.feature_attribution(np.expand_dims(x[i], axis=0))
                    else:
                        fs2 = explainer2.feature_attribution(np.expand_dims(x[i], axis=0))
                    fs_mid.append(fs2)
                    score_mid.append(explainer2.score(samples, labels, sample_weight=weights))
                    rule_mid.append(explainer2.num_rule)
                    rem_pre_mid.append(rem_num2)
                    pass
                j += 1

            if j != 0:
                self.explainer.lse_con_param(samples, labels, weights)
                tra_loss, _, membership_value = calculate_error(samples, labels, self.explainer, weight=weights)
                if self.mode == 'regression':
                    fs = self.explainer.feature_attribution(np.expand_dims(x[i], axis=0))
                else:
                    fs = self.explainer.feature_attribution(np.expand_dims(x[i], axis=0))
                fs_hier.append(fs)
                log_hier.append(self.explainer.num_rule)
            rule_end.append(self.explainer.num_rule)

            # premise condensation
            if not self.explainer.is_linear:
                self.explainer, rem_num = premise_condensation(self.explainer, membership_value, samples, labels,
                                                               tra_loss, tra_loss_ini, tau_R=1.1, log=True,
                                                               weight=weights)
                if rem_num != 0:
                    self.explainer.lse_con_param(samples, labels, weights)
                    if self.mode == 'regression':
                        fs = self.explainer.feature_attribution(np.expand_dims(x[i], axis=0))
                    else:
                        fs = self.explainer.feature_attribution(np.expand_dims(x[i], axis=0))
                    fs_hier.append(fs)
                    log_hier.append(rem_num)
                else:
                    log_hier.append(0)
            else:
                log_hier.append(-1)
                rem_num = 0

            fs_end.append(fs)
            score_end.append(self.explainer.score(samples, labels, sample_weight=weights))
            rem_pre_end.append(rem_num)

            fs_all.append(fs_hier)
            log_all.append(log_hier)
            zoomer_all.append(zoomer)

        self.zoomer = np.mean(zoomer_all)  # zoomer
        self.score = np.average(score_end)
        ini_ = [np.array(fs_ini), np.array(fs_end), score_ini, score_end, rule_ini, rule_end, rem_pre_end]
        mid_ = [np.array(fs_mid), score_mid, rule_mid, rem_pre_mid]
        if self.mid_info:
            return fs_all, log_all, ini_, mid_
        else:
            return fs_end
