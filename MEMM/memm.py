import itertools
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import sklearn
from numpy.linalg import norm
import numpy as np
import pickle
import os
import scipy.optimize
import pandas as pd
pd.set_option('display.max_columns', None)


class Mmem:

    def __init__(self, la, threshold, comp=False, k=None):
        self.likl_func = []
        self.likl_grad = []
        self.tag_set = [0, 1]
        self.histories = []
        self.n_total_features = 0  # Total number of features accumulated
        self.features_list = ['f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107',
                                    'f108', 'f109', 'f110', 'f111',
                                      'f201', 'f202', 'f203', 'f204', 'f205', 'f206', 'f207',
                                    'f208', 'f209', 'f210', 'f211',]
        self.features_count = {f_name: {} for f_name in self.features_list}
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this
        self.la = la
        self.B = 7
        self.hyper_str = str(self.threshold) + '_' + str(self.la)
        self.model_dir = '/home/student/Desktop/ML/MEMM/saves/'.format()
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.n_dict = {}
        self.features_id_dict = {f_name: {} for f_name in self.features_list}  # OrderedDict
        self.train_features = {}  # (x_i, y_): on_features_list
        self.all_tags_features = {}  # (x_i, y_): on_features_list
        self.train_vector_sum = np.zeros(self.n_total_features)
        self.all_tags_exp = {}
        self.w = np.zeros(self.n_total_features)
        self.minimization_dict = {}
        self.accuracy = 0
        self.skip_cols = ['Airport_Code', 'Unnamed: 0', 'Severity']
        self.bars_dict = {}
        self.cols_dict = {}
        self.minmax = {}
        self.markov = 5

    @staticmethod
    def tup_of_tups(df):
        return tuple(df.itertuples(index=False, name=None))

    @staticmethod
    def create_bars(df, col):
        min_val = df['Temperature(F)'].quantile(.025)
        max_val = df['Temperature(F)'].quantile(.975)
        return [min_val + i * (max_val - min_val) / 20 for i in range(21)]

    @staticmethod
    def get_bar(bars, val):
        i = 0
        bars_len = len(bars)
        while i < bars_len and bars[i] < val:
            i += 1
        return i

    def add2count(self, key, dictionary):
        """"add 1 to the key in the dictionary (or create the key with value 1)."""
        if key not in self.features_count[dictionary]:
            self.features_count[dictionary][key] = 1
        else:
            self.features_count[dictionary][key] += 1

    def add2count_cont(self, df, row_i, curr_severity, col, code):
        bars = self.create_bars(df, col)
        bar_i = self.get_bar(bars, row_i[col])
        self.bars_dict[col] = bars
        self.add2count((bar_i, curr_severity), code)
        return bar_i

    def create_features(self):
        """defines and create the features, counting number of occurence of each one."""
        df = pd.read_csv('/home/student/Desktop/ML/save.csv')
        print('len', len(df))

        grouped = df.groupby('City')
        print('num of groups', len(grouped))
        for group_id, (group, df_group) in enumerate(grouped):
            print(group_id, len(df_group), end=', ')
            for i in range(self.markov):
                df_group.loc[-i] = ['*'] * len(df_group.columns)  # adding a row
            df_group.index = df_group.index + self.markov  # shifting index
            df_group = df_group.sort_index().reset_index(drop=True)
            # print('df_group', df_group.head(12))
            for i in range(self.markov, len(df_group)):  # iterate over indices of words in sentence
                row_i = df_group.iloc[i]
                history_df = df_group.iloc[i - self.markov:i + 1].reset_index(drop=True)
                curr_severity = df_group.loc[i, 'Severity']
                self.cols_dict = dict(zip(df.columns, range(len(df))))
                self.histories.append((history_df, curr_severity))  # add to histories this history and tag

                self.add2count((row_i['Traffic_Signal'], curr_severity), 'f101')

                self.add2count((row_i['Crossing'], curr_severity), 'f102')

                self.add2count((row_i['Junction'], curr_severity), 'f103')

                self.add2count((row_i['Station'], curr_severity), 'f104')

                self.add2count((row_i['Stop'], curr_severity), 'f105')

                temp_bar_i = self.add2count_cont(df, row_i, curr_severity, 'Temperature(F)', 'f106')

                dist_bar_i = self.add2count_cont(df, row_i, curr_severity, 'Distance(mi)', 'f107')

                humi_bar_i = self.add2count_cont(df, row_i, curr_severity, 'Humidity(%)', 'f108')

                wind_bar_i = self.add2count_cont(df, row_i, curr_severity, 'Wind_Speed(mph)', 'f109')

                visi_bar_i = self.add2count_cont(df, row_i, curr_severity, 'Visibility(mi)', 'f110')

                pres_bar_i = self.add2count_cont(df, row_i, curr_severity, 'Pressure(in)', 'f111')

                for j in range(self.markov):  # TODO: maybe range(1, 11)
                    if history_df.loc[j, 'Severity'] == '*':
                        continue
                    self.add2count((history_df.loc[j, 'Severity'], j, curr_severity), 'f100')

                    self.add2count((history_df.loc[j, 'Traffic_Signal'], j, curr_severity), 'f201')

                    self.add2count((history_df.loc[j, 'Crossing'], j, curr_severity), 'f202')

                    self.add2count((history_df.loc[j, 'Junction'], j, curr_severity), 'f203')

                    self.add2count((history_df.loc[j, 'Station'], j, curr_severity), 'f204')

                    self.add2count((history_df.loc[j, 'Stop'], j, curr_severity), 'f205')

                    temp_bar_j = self.get_bar(self.bars_dict['Temperature(F)'], history_df.loc[j, 'Temperature(F)'])
                    self.add2count((temp_bar_i, temp_bar_j, j, curr_severity), 'f206')

                    dist_bar_j = self.get_bar(self.bars_dict['Distance(mi)'], history_df.loc[j, 'Distance(mi)'])
                    self.add2count((dist_bar_i, dist_bar_j, j, curr_severity), 'f207')

                    humi_bar_j = self.get_bar(self.bars_dict['Humidity(%)'], history_df.loc[j, 'Humidity(%)'])
                    self.add2count((humi_bar_i, humi_bar_j, j, curr_severity), 'f208')

                    wind_bar_j = self.get_bar(self.bars_dict['Wind_Speed(mph)'], history_df.loc[j, 'Wind_Speed(mph)'])
                    self.add2count((wind_bar_i, wind_bar_j, j, curr_severity), 'f209')

                    visi_bar_j = self.get_bar(self.bars_dict['Visibility(mi)'], history_df.loc[j, 'Visibility(mi)'])
                    self.add2count((visi_bar_i, visi_bar_j, j, curr_severity), 'f210')

                    pres_bar_j = self.get_bar(self.bars_dict['Pressure(in)'], history_df.loc[j, 'Pressure(in)'])
                    self.add2count((pres_bar_i, pres_bar_j, j, curr_severity), 'f211')

        # print('self.features_count', self.features_count)
        print('finished groups')

    def preprocess_features(self):
        """filter features that occured in train set less than threshold,
        and gives an ID for each one."""
        for feature_name, feature_dict in self.features_count.items():
            self.n_dict[feature_name] = self.n_total_features
            for key, count in feature_dict.items():
                if count >= self.threshold:
                    self.features_id_dict[feature_name][key] = self.n_dict[feature_name]
                    self.n_dict[feature_name] += 1
            self.n_total_features = self.n_dict[feature_name]
            print(feature_name, self.n_total_features)
        print(self.features_id_dict)

    def history2features(self, history_df, curr_severity):
        """return a list of features ID given a history"""
        features = []
        # adds the id of a feature if its conditions happens on the given history and tag
        row_i = history_df.iloc[-1]

        key = (row_i['Traffic_Signal'], curr_severity)
        if key in self.features_id_dict['f101']:
            features.append(self.features_id_dict['f101'][key])

        key = (row_i['Crossing'], curr_severity)
        if key in self.features_id_dict['f102']:
            features.append(self.features_id_dict['f102'][key])

        key = (row_i['Junction'], curr_severity)
        if key in self.features_id_dict['f103']:
            features.append(self.features_id_dict['f103'][key])

        key = (row_i['Station'], curr_severity)
        if key in self.features_id_dict['f104']:
            features.append(self.features_id_dict['f104'][key])

        key = (row_i['Stop'], curr_severity)
        if key in self.features_id_dict['f105']:
            features.append(self.features_id_dict['f105'][key])

        temp_bar_i = self.get_bar(self.bars_dict['Temperature(F)'], row_i['Temperature(F)'])
        key = (temp_bar_i, curr_severity)
        if key in self.features_id_dict['f106']:
            features.append(self.features_id_dict['f106'][key])

        dist_bar_i = self.get_bar(self.bars_dict['Distance(mi)'], row_i['Distance(mi)'])
        key = (dist_bar_i, curr_severity)
        if key in self.features_id_dict['f107']:
            features.append(self.features_id_dict['f107'][key])

        humi_bar_i = self.get_bar(self.bars_dict['Humidity(%)'], row_i['Humidity(%)'])
        key = (humi_bar_i, curr_severity)
        if key in self.features_id_dict['f108']:
            features.append(self.features_id_dict['f108'][key])

        wind_bar_i = self.get_bar(self.bars_dict['Wind_Speed(mph)'], row_i['Distance(mi)'])
        key = (wind_bar_i, curr_severity)
        if key in self.features_id_dict['f109']:
            features.append(self.features_id_dict['f109'][key])

        visi_bar_i = self.get_bar(self.bars_dict['Visibility(mi)'], row_i['Visibility(mi)'])
        key = (visi_bar_i, curr_severity)
        if key in self.features_id_dict['f110']:
            features.append(self.features_id_dict['f110'][key])

        pres_bar_i = self.get_bar(self.bars_dict['Pressure(in)'], row_i['Pressure(in)'])
        key = (pres_bar_i, curr_severity)
        if key in self.features_id_dict['f111']:
            features.append(self.features_id_dict['f111'][key])

        for j in range(self.markov):
            if history_df.loc[j, 'Severity'] == '*':
                continue

            key = (history_df.loc[j, 'Severity'], j, curr_severity)
            if key in self.features_id_dict['f100']:
                features.append(self.features_id_dict['f100'][key])

            key = (history_df.loc[j, 'Traffic_Signal'], j, curr_severity)
            if key in self.features_id_dict['f201']:
                features.append(self.features_id_dict['f201'][key])

            key = (history_df.loc[j, 'Crossing'], j, curr_severity)
            if key in self.features_id_dict['f202']:
                features.append(self.features_id_dict['f202'][key])

            key = (history_df.loc[j, 'Junction'], j, curr_severity)
            if key in self.features_id_dict['f203']:
                features.append(self.features_id_dict['f203'][key])

            key = (history_df.loc[j, 'Station'], j, curr_severity)
            if key in self.features_id_dict['f204']:
                features.append(self.features_id_dict['f204'][key])

            key = (history_df.loc[j, 'Stop'], j, curr_severity)
            if key in self.features_id_dict['f205']:
                features.append(self.features_id_dict['f205'][key])

            temp_bar_j = self.get_bar(self.bars_dict['Temperature(F)'], history_df.iloc[j]['Temperature(F)'])
            key = (temp_bar_i, temp_bar_j, j, curr_severity)
            if key in self.features_id_dict['f206']:
                features.append(self.features_id_dict['f206'][key])

            dist_bar_j = self.get_bar(self.bars_dict['Distance(mi)'], history_df.iloc[j]['Distance(mi)'])
            key = (dist_bar_i, dist_bar_j, j, curr_severity)
            if key in self.features_id_dict['f207']:
                features.append(self.features_id_dict['f207'][key])

            humi_bar_j = self.get_bar(self.bars_dict['Humidity(%)'], history_df.iloc[j]['Humidity(%)'])
            key = (humi_bar_i, humi_bar_j, j, curr_severity)
            if key in self.features_id_dict['f208']:
                features.append(self.features_id_dict['f208'][key])

            wind_bar_j = self.get_bar(self.bars_dict['Wind_Speed(mph)'], history_df.iloc[j]['Wind_Speed(mph)'])
            key = (wind_bar_i, wind_bar_j, j, curr_severity)
            if key in self.features_id_dict['f209']:
                features.append(self.features_id_dict['f209'][key])

            visi_bar_j = self.get_bar(self.bars_dict['Visibility(mi)'], history_df.iloc[j]['Visibility(mi)'])
            key = (visi_bar_i, visi_bar_j, j, curr_severity)
            if key in self.features_id_dict['f210']:
                features.append(self.features_id_dict['f210'][key])

            pres_bar_j = self.get_bar(self.bars_dict['Pressure(in)'], history_df.iloc[j]['Pressure(in)'])
            key = (pres_bar_i, pres_bar_j, j, curr_severity)
            if key in self.features_id_dict['f211']:
                features.append(self.features_id_dict['f211'][key])

        return features

    def extract_features(self):
        i = 0
        for history, cur_tag in self.histories:
            if i % 500 == 0:
                print(i, end=', ')
            i += 1
            curr_features = self.history2features(history, cur_tag)
            self.train_features[(self.tup_of_tups(history), cur_tag)] = curr_features
            self.all_tags_features[(self.tup_of_tups(history), cur_tag)] = curr_features
            self.all_tags_features[(self.tup_of_tups(history), 1 - cur_tag)] = self.history2features(history, 1 - cur_tag)

    def linear_term(self, w):
        """calculate the linear term of the likelihood function: sum for each i: w*f(x_i,y_i)"""
        linear_sum = 0
        for feature_dict in self.train_features.values():
            # linear_sum += (w[list(feature_dict.keys())] * list(feature_dict.values())).sum()
            linear_sum += w[feature_dict].sum()
        return linear_sum

    def normalization_term(self, w):
        all_sum = 0
        for history, _ in self.histories:
            log_sum = 0
            for y_ in self.tag_set:
                feature_dict = self.all_tags_features[(self.tup_of_tups(history), y_)]
                # log_sum += np.exp((w[list(feature_dict.keys())] * list(feature_dict.values())).sum())
                log_sum += np.exp(w[feature_dict].sum())
            self.all_tags_exp[self.tup_of_tups(history)] = log_sum
            all_sum += np.log(log_sum)
        return all_sum

    def empirical_counts(self):
        self.train_vector_sum = np.zeros(self.n_total_features)
        for feature_dict in self.train_features.values():
            # self.train_vector_sum[list(feature_dict.keys())] += list(feature_dict.values())
            self.train_vector_sum[feature_dict] += 1

    def expected_counts(self, w):
        """calculate the expected term of the loss function"""
        all_sum = np.zeros(self.n_total_features)
        for history, cur_tag in self.histories:
            deno = self.all_tags_exp[self.tup_of_tups(history)]
            # print('deno', deno)
            # time.sleep(3)
            for y_ in self.tag_set:
                feature_dict = self.all_tags_features[(self.tup_of_tups(history), y_)]
                # nom = np.exp((w[list(feature_dict.keys())] * list(feature_dict.values())).sum())
                nom = np.exp(w[feature_dict].sum())
                all_sum[self.all_tags_features[(self.tup_of_tups(history), y_)]] += (nom / deno)
        return all_sum

    def func(self, w):
        print('function run', w.sum())
        t0 = time.time()
        linear_term = self.linear_term(w)
        t1 = time.time()
        normalization_term = self.normalization_term(w)
        t2 = time.time()

        print('linear_term', t1-t0, linear_term)
        print('normalization_term', t2-t1, normalization_term)
        likelihood = linear_term - normalization_term - 0.5 * self.la * (np.linalg.norm(w) ** 2)
        return -likelihood

    def f_prime(self, w):
        """calculates the likelihood function and its gradient."""
        print('gradient run', w.sum())
        t2 = time.time()
        expected_counts = self.expected_counts(w)
        t3 = time.time()

        print('expected_counts', t3-t2, expected_counts[-10:])
        grad = self.train_vector_sum - expected_counts - self.la * w
        return -grad

    def minimize(self):
        """applies the minimization for the loss function"""
        self.empirical_counts()
        w = np.random.rand(self.n_total_features) / 100 - 0.005  # initiates the weigths to small values
        w, _, _ = scipy.optimize.fmin_l_bfgs_b(func=self.func, fprime=self.f_prime,
                                               x0=w, maxiter=15, epsilon=10 ** (-6), iprint=1)
        # w, _, _ = scipy.optimize.fmin_bfgs(f=self.func, fprime=self.f_prime,
        #                                    x0=w, maxiter=15, epsilon=10 ** (-6), iprint=1)

        self.w = w
        print('self.w', self.w)

    def save_model(self):
        """saves the train result to be able to test using them"""
        with open(self.model_dir + 'weights.pkl', 'wb') as file:
            pickle.dump(self.w, file)
        with open(self.model_dir + 'features_id_dict.pkl', 'wb') as file:
            pickle.dump(self.features_id_dict, file)
        with open(self.model_dir + 'bars.pkl', 'wb') as file:
            pickle.dump(self.bars_dict, file)

    def train(self, train_path=None):
        """train the model"""
        print('Beginning train')

        self.create_features()
        print('created features')
        self.preprocess_features()
        print('preprocessed features')
        self.extract_features()
        print('extracted features')
        self.minimize()
        print('minimized features')
        self.save_model()

    def test(self, run_train=True):
        """test the model"""
        test_path = 'data/train.csv'
        print("Beginning test on", test_path)

        if not run_train:
            with open(self.model_dir + 'weights.pkl', 'rb') as file:
                self.w = pickle.load(file)
            with open(self.model_dir + 'features_id_dict.pkl', 'rb') as file:
                self.features_id_dict = pickle.load(file)
            with open(self.model_dir + 'bars.pkl', 'rb') as file:
                self.bars_dict = pickle.load(file)

        print(self.features_id_dict.keys())
        all_sentences = []  # list of sentences for saving predicted tags in competition
        all_t_tags = []  # list of true tags for comparing on test
        all_p_tags = []  # list of predicted tags
        df = pd.read_csv('/home/student/Desktop/ML/save1.csv')

        grouped = df.groupby('Airport_Code')
        for group_id, (group, df_group) in enumerate(grouped):
            print(group_id, len(df_group), end=', ')
            # if len(df_group) < 20:
            #     continue
            all_t_tags.append(df_group)
            p_tags = self.viterbi_beam(df_group)
            print(df_group['Severity'])
            all_p_tags.append(p_tags)
        self.evaluate(all_t_tags, all_p_tags)

    def evaluate(self, all_t_tags, all_p_tags):
        """calculates the accuracy and confusion matrix for prediction"""
        all_p_tags = [item for sublist in all_p_tags for item in sublist]
        all_t_tags = [item for sublist in all_t_tags for item in sublist]

        results = precision_recall_fscore_support(all_t_tags, all_p_tags, average='binary')
        results = [accuracy_score(all_t_tags, all_p_tags)] + list(results)
        results = str(dict(zip(['accuracy', 'precision', 'recall', 'f1'], results)))
        print(results)
        with open(self.model_dir + "accuracy.txt", 'w') as file:
            file.write(results)
        # with open(self.model_dir + "percent_cm.txt", 'w') as file:
        #     file.write(percent_cm)
        # with open(self.model_dir + "count_cm.txt", 'w') as file:
        #     file.write(count_cm)

    def pi_q(self, pi, history_df, k, t, x):
        """calculate pi"""
        history_df['Severity'] = [t, ] + list(x)
        feature_dict = self.history2features(history_df, x[-1])
        nome = np.exp(self.w[feature_dict].sum())
        feature_dict = self.history2features(history_df, 1 - x[-1])
        deno = np.exp(self.w[feature_dict].sum()) + nome

        return pi[(k - 1, (t, ) + x[:-1])] * nome / deno

    def viterbi_beam_old(self, sentence):
        """perform viterbi with beam-search heuristics"""
        pi = {}
        bp = {}
        pi[(0, '*', '*')] = 1
        s_1 = ['*']
        s_2 = ['*']
        for k in range(1, len(sentence) + 1):
            if k == 2:
                s_1 = self.tag_set
                s_2 = ['*']
            b_best_pi = {}
            for u in s_1:
                for v in self.tag_set:
                    bp[(k, u, v)] = bp_calc = max(s_2, key=lambda t: self.pi_q(pi, sentence, k, t, u, v))
                    pi[(k, u, v)] = pi_calc = self.pi_q(pi, sentence, k, bp_calc, u, v)

                    if len(b_best_pi) < self.B:
                        b_best_pi[(u, v)] = pi_calc
                    else:
                        min_key, min_val = min(b_best_pi.items(), key=lambda x: x[1])
                        if pi_calc > min_val:
                            b_best_pi[(u, v)] = pi_calc
                            del b_best_pi[min_key]
            if k != len(sentence):
                s_1, s_2 = [], []
                for key in b_best_pi.keys():
                    s_2.append(key[0])
                    s_1.append(key[1])
        tag_list = list(
            max([(u, v) for u in s_1 for v in self.tag_set], key=lambda uv: pi[(len(sentence), uv[0], uv[1])]))
        for k in reversed(range(1, len(sentence) - 1)):
            tag_list = [bp[(k + 2, tag_list[0], tag_list[1])]] + tag_list
        return tag_list

    def pi_q_beam(self, pi, history_df, k, t, x):
        """calculate pi"""
        if (k - 1, (t, ) + x[:-1]) not in pi:
            return -1
        history_df['Severity'] = [t, ] + list(x)
        feature_dict = self.history2features(history_df, x[-1])
        nome = np.exp(self.w[feature_dict].sum())
        feature_dict = self.history2features(history_df, 1 - x[-1])
        deno = np.exp(self.w[feature_dict].sum()) + nome

        return pi[(k - 1, (t, ) + x[:-1])] * nome / deno

    def viterbi_beam(self, df_group):
        """perform viterbi"""
        print('len', len(df_group))
        for i in range(self.markov):
            df_group.loc[-i] = ['*'] * len(df_group.columns)  # adding a row
        df_group.index = df_group.index + self.markov  # shifting index
        df_group = df_group.sort_index().reset_index(drop=True)

        pi = {}
        bp = {}
        pi[self.markov - 1, ('*',) * self.markov] = 1
        for k in range(self.markov, len(df_group)):
            print(k, end=', ')
            history_df = df_group.iloc[k - self.markov:k + 1].reset_index(drop=True)
            curr_severity = df_group.loc[k, 'Severity']
            b_best_pi = {}
            pi_k = {}
            bp_k = {}

            if k < self.markov*2:
                ss = [['*']] * (self.markov*2 - k) + [self.tag_set] * (k - self.markov + 1)
            else:
                ss = [self.tag_set] * (self.markov + 1)

            for x in itertools.product(*ss[1:]):
                bp_k[(k, x)] = max(ss[0], key=lambda t: self.pi_q_beam(pi, history_df, k, t, x))
                pi_k[(k, x)] = pi_calc = self.pi_q_beam(pi, history_df, k, bp_k[(k, x)], x)

                if len(b_best_pi) < self.B:
                    b_best_pi[x] = pi_calc
                else:
                    min_key, min_val = min(b_best_pi.items(), key=lambda a: a[1])
                    if pi_calc > min_val:
                        b_best_pi[x] = pi_calc
                        del b_best_pi[min_key]
                pi = {**pi, **{(k, x): pi_k[(k, x)] for x in b_best_pi.keys()}}
                bp = {**bp, **{(k, x): bp_k[(k, x)] for x in b_best_pi.keys()}}

        t = list(max([x for x in itertools.product(*ss[1:])],
                 key=lambda x: self.get_pi(pi, len(df_group)-1, x)))
        for k in reversed(range(self.markov, len(df_group))):
            t = [bp[(k, tuple(t[:self.markov]))]] + t
        print(t)
        return t

    def get_pi(self, pi, k, x):
        if (k, x) not in pi:
            return -1
        return pi[(k, x)]

    def viterbi(self, df_group):
        """perform viterbi"""
        print('len', len(df_group))
        for i in range(self.markov):
            df_group.loc[-i] = ['*'] * len(df_group.columns)  # adding a row
        df_group.index = df_group.index + self.markov  # shifting index
        df_group = df_group.sort_index().reset_index(drop=True)

        pi = {}
        bp = {}
        pi[self.markov - 1, ('*',) * self.markov] = 1
        for k in range(self.markov, len(df_group)):
            print(k, end=', ')
            history_df = df_group.iloc[k - self.markov:k + 1].reset_index(drop=True)
            curr_severity = df_group.loc[k, 'Severity']

            if k < self.markov*2:
                ss = [['*']] * (self.markov*2 - k) + [self.tag_set] * (k - self.markov - 1)
            else:
                ss = [self.tag_set] * self.markov + 1

            for x in itertools.product(*ss[1:]):
                bp[(k, x)] = max(ss[0], key=lambda t: self.pi_q(pi, history_df, k, t, x))
                pi[(k, x)] = self.pi_q(pi, history_df, k, bp[(k, x)], x)
        # print(pi)
        t = list(max([x for x in itertools.product(*ss[1:])],
                 key=lambda x: pi[(len(df_group) - 1, x)]))
        for k in reversed(range(self.markov, len(df_group))):
            t = [bp[(k, tuple(t[:self.markov]))]] + t
        print(t)
        return t


if __name__ == '__main__':
    print('Welcome to our accidents predictor!')
    # while True:
    #     run_train = input("Should we train before testing?[y/n]   ")
    #     if run_train.lower() in ['y', 'n', 'yes', 'no']:
    #         run_train = True if run_train in ['y', 'yes'] else False
    #         break
    #     print("invalid input. please enter 'y' or 'n'")

    run_train = True
    # run_train = False
    thre = 1
    la = 0.1

    model = Mmem(la=la, threshold=thre)
    if run_train:
        model.train()
    model.test(run_train=run_train)
