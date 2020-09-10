import itertools
import time
from sklearn.metrics import mean_squared_error, accuracy_score
from numpy.linalg import norm
import numpy as np
import pickle
import os
import scipy.optimize
import pandas as pd

pd.set_option('display.max_columns', None)


class Memm:

    def __init__(self, model_name):
        self.model_name = model_name
        self.train_path = '/home/student/Desktop/ML/weather_data/Sequence_data_frames/df_{}_train.csv'.format(self.model_name)
        self.test_path = '/home/student/Desktop/ML/weather_data/Sequence_data_frames/df_{}_test.csv'.format(self.model_name)
        self.model_dir = '/home/student/Desktop/ML/MEMM/saves/{}'.format(self.model_name)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.likl_func = []
        self.likl_grad = []
        self.tag_set = list(range(11))
        self.histories = []
        self.n_total_features = 0  # Total number of features accumulated

        self.cate = [' _conds', ' _fog', ' _hail', ' _rain', ' _snow', ' _thunder', ' _tornado', 'night', 'morning',
                     'noon', 'evening', 'year', 'month', 'week', 'day', 'hour', ' _dewptm_bars', ' _hum_bars',
                     ' _pressurem_bars', ' _vism_bars', ' _wspdm_bars']
        self.cate_map = [(name, str(i).rjust(2, '0')) for i, name in enumerate(self.cate)]

        self.cont = []  # [' _dewptm', ' _pressurem', ' _vism', ' _wspdm',
        #              'month_cos', 'month_sin', 'hour_cos', 'hour_sin', 'week_cos', 'week_sin']
        self.cont_map = [(name, str(i).rjust(2, '2')) for i, name in enumerate(self.cont)]

        self.features_codes = ['f99'] + ['f' + i + c[1] for c in self.cate_map + self.cont_map for i in ['1', '2']]
        self.features_count = {f_name: {} for f_name in self.features_codes}
        self.col_map = {}

        self.n_dict = {}
        self.features_id_dict = {f_name: {} for f_name in self.features_codes}  # OrderedDict
        self.train_features = {}
        self.all_tags_features = {}
        self.train_vector_sum = np.zeros(self.n_total_features)
        self.all_tags_exp = {}

        self.w = np.zeros(self.n_total_features)
        self.minimization_dict = {}
        self.bars_dict = {}
        self.cols_dict = {}

        self.markov = 2
        self.threshold = {f_name: 0 for f_name in
                          self.features_codes}  # feature count threshold - empirical count must be higher than this
        self.la = 0.1
        self.B = 7
        self.hyper_str = str(self.threshold) + '_' + str(self.la)

    @staticmethod
    def tup_of_tups(history_list):
        return tuple(tuple(x) for x in history_list)

    @staticmethod
    def create_bars(df, col_name):
        min_val = df[col_name].quantile(.025)
        max_val = df[col_name].quantile(.975)
        return [min_val + i * (max_val - min_val) / 20 for i in range(21)]

    @staticmethod
    def get_bar(bars, val):
        i = 0
        bars_len = len(bars)
        while i < bars_len and bars[i] < val:
            i += 1
        return i

    def add_stars(self, df):
        df_star = pd.DataFrame([['*'] * len(df.columns)] * self.markov, columns=df.columns)
        return pd.concat([df_star, df]).reset_index(drop=True)

    def add2count(self, key, dictionary):
        """"add 1 to the key in the dictionary (or create the key with value 1)."""
        if key not in self.features_count[dictionary]:
            self.features_count[dictionary][key] = 1
        else:
            self.features_count[dictionary][key] += 1

    def add2count_cont(self, df, row_i, curr_temp, col_name, code):
        bars = self.create_bars(df, col_name)
        bar_i = self.get_bar(bars, row_i[self.col_map[col_name]])
        self.bars_dict[col_name] = bars
        self.add2count((bar_i, curr_temp), code)
        return bar_i

    def create_features(self):
        """defines and create the features, counting number of occurence of each one."""
        df = pd.read_csv(self.train_path).head(1000)
        self.col_map = {name: i for i, name in enumerate(df.columns)}

        grouped = df.groupby(['sequence'])
        print('len =', len(grouped))
        last_group = {'year': None, 'month': None}
        for group_id, (group, df_seq) in enumerate(grouped):
            curr_group = df_seq[['year', 'month']].iloc[-1].to_dict()
            if group_id == 0 or \
                    ('full' not in self.model_name and 'winter' not in self.model_name and
                     last_group['year'] != curr_group['year']) or \
                    ('winter' in self.model_name and curr_group['month'] != '1'):
                df_seq = self.add_stars(df_seq)
            df_seq = df_seq.values.tolist()
            for i in range(self.markov, len(df_seq)):  # iterate over indices of words in sentence
                row_i = df_seq[i]
                history_list = df_seq[i - self.markov:i + 1]
                curr_temp = row_i[self.col_map['Temp']]
                self.histories.append((history_list, curr_temp))  # add to histories this history and tag

                # categorical
                for col_name, code in self.cate_map:
                    self.add2count((row_i[self.col_map[col_name]], curr_temp), 'f1' + code)

                bars_i = {}
                # continuous
                for col_name, code in self.cont_map:
                    bars_i[col_name] = self.add2count_cont(df, row_i, curr_temp, col_name, 'f1' + code)

                for j in range(self.markov):  # TODO: maybe range(1, self.markov)
                    row_j = history_list[j]
                    if row_j[self.col_map['Temp']] == '*':
                        continue
                    self.add2count((row_j[self.col_map['Temp']], j, curr_temp), 'f100')

                    # categorical
                    for col_name, code in self.cate_map:
                        self.add2count((row_j[self.col_map[col_name]], j, curr_temp), 'f2' + code)

                    # continuous
                    for col_name, code in self.cont_map:
                        temp_bar_j = self.get_bar(self.bars_dict[col_name], row_j[self.col_map[col_name]])
                        self.add2count((bars_i[col_name], temp_bar_j, j, curr_temp), 'f2' + code)
            last_group = curr_group

    def preprocess_features(self):
        """filter features that occured in train set less than threshold,
        and gives an ID for each one."""
        prev_n = 0
        for feature_code, feature_dict in self.features_count.items():
            self.n_dict[feature_code] = self.n_total_features
            for key, count in feature_dict.items():
                if count >= self.threshold[feature_code]:
                    self.features_id_dict[feature_code][key] = self.n_dict[feature_code]
                    self.n_dict[feature_code] += 1
            self.n_total_features = self.n_dict[feature_code]
            # print(feature_code, self.n_total_features - prev_n, end=', ')
            prev_n = self.n_total_features

    def history2features(self, history_list, curr_temp):
        """return a list of features ID given a history"""
        features = []
        row_i = history_list[-1]

        # categorical
        for col_name, code in self.cate_map:
            key = (row_i[self.col_map[col_name]], curr_temp)
            if key in self.features_id_dict['f1' + code]:
                features.append(self.features_id_dict['f1' + code][key])
        bars_i = {}
        # continuous
        for col_name, code in self.cont_map:
            bars_i[col_name] = self.get_bar(self.bars_dict[col_name], row_i[self.col_map[col_name]])
            key = (bars_i[col_name], curr_temp)
            if key in self.features_id_dict['f1' + code]:
                features.append(self.features_id_dict['f1' + code][key])

        for j in range(self.markov):
            row_j = history_list[j]
            if row_j[self.col_map['Temp']] == '*':
                continue

            # categorical
            for col_name, code in self.cate_map:
                key = (row_j[self.col_map[col_name]], j, curr_temp)
                if key in self.features_id_dict['f1' + code]:
                    features.append(self.features_id_dict['f1' + code][key])

            # continuous
            for col_name, code in self.cont_map:
                bar_j = self.get_bar(self.bars_dict[col_name], row_j[self.col_map[col_name]])
                key = (bars_i[col_name], bar_j, j, curr_temp)
                if key in self.features_id_dict['f2' + code]:
                    features.append(self.features_id_dict['f2' + code][key])

        return features

    def extract_features(self):
        i = 0
        for history, cur_tag in self.histories:
            # if i % 500 == 0:
            #     print(i, end=', ')
            i += 1
            curr_features = self.history2features(history, cur_tag)
            self.train_features[(self.tup_of_tups(history), cur_tag)] = curr_features
            self.all_tags_features[(self.tup_of_tups(history), cur_tag)] = curr_features
            for y_ in self.tag_set:
                if y_ == cur_tag:
                    continue
                self.all_tags_features[(self.tup_of_tups(history), y_)] = self.history2features(history, y_)

    def linear_term(self, w):
        """calculate the linear term of the likelihood function: sum for each i: w*f(x_i,y_i)"""
        linear_sum = 0
        for feature_dict in self.train_features.values():
            linear_sum += w[feature_dict].sum()
        return linear_sum

    def normalization_term(self, w):
        all_sum = 0
        for history, _ in self.histories:
            log_sum = 0
            for y_ in self.tag_set:
                feature_dict = self.all_tags_features[(self.tup_of_tups(history), y_)]
                log_sum += np.exp(w[feature_dict].sum())
            self.all_tags_exp[self.tup_of_tups(history)] = log_sum
            all_sum += np.log(log_sum)
        return all_sum

    def empirical_counts(self):
        self.train_vector_sum = np.zeros(self.n_total_features)
        for feature_dict in self.train_features.values():
            self.train_vector_sum[feature_dict] += 1

    def expected_counts(self, w):
        """calculate the expected term of the loss function"""
        all_sum = np.zeros(self.n_total_features)
        for history, cur_tag in self.histories:
            deno = self.all_tags_exp[self.tup_of_tups(history)]
            for y_ in self.tag_set:
                feature_dict = self.all_tags_features[(self.tup_of_tups(history), y_)]
                nom = np.exp(w[feature_dict].sum())
                all_sum[self.all_tags_features[(self.tup_of_tups(history), y_)]] += (nom / deno)
        return all_sum

    def func(self, w):
        # print('function run', w.sum())
        t0 = time.time()
        linear_term = self.linear_term(w)
        t1 = time.time()
        normalization_term = self.normalization_term(w)
        t2 = time.time()

        # print('linear_term', t1 - t0, linear_term)
        # print('normalization_term', t2 - t1, normalization_term)
        likelihood = linear_term - normalization_term - 0.5 * self.la * (np.linalg.norm(w) ** 2)

        self.likl_func.append(-likelihood)
        return -likelihood

    def f_prime(self, w):
        """calculates the likelihood function and its gradient."""
        # print('gradient run', w.sum())
        t2 = time.time()
        expected_counts = self.expected_counts(w)
        t3 = time.time()

        # print('expected_counts', t3 - t2, expected_counts[-10:])
        grad = self.train_vector_sum - expected_counts - self.la * w

        self.likl_grad.append(np.linalg.norm(-grad))
        return -grad

    def minimize(self):
        """applies the minimization for the loss function"""
        self.empirical_counts()
        w = np.random.rand(self.n_total_features) / 100 - 0.005
        w, _, _ = scipy.optimize.fmin_l_bfgs_b(func=self.func, fprime=self.f_prime,
                                               x0=w, maxiter=100, epsilon=10 ** (-6), iprint=0)

        self.w = w
        # print('self.w', self.w)

    def train(self):
        """train the model"""
        print('Beginning train...')

        self.create_features()
        # print('created features')
        self.preprocess_features()
        # print('preprocessed features')
        self.extract_features()
        # print('extracted features')
        self.minimize()
        # print('minimized features')
        self.save_model()

    def save_model(self):
        """saves the train result to be able to test using them"""
        with open(os.path.join(self.model_dir, 'weights.pkl'), 'wb') as file:
            pickle.dump(self.w, file)
        with open(os.path.join(self.model_dir, 'features_id_dict.pkl'), 'wb') as file:
            pickle.dump(self.features_id_dict, file)
        with open(os.path.join(self.model_dir, 'bars.pkl'), 'wb') as file:
            pickle.dump(self.bars_dict, file)

        with open(os.path.join(self.model_dir, 'mini.txt'), 'w') as file:
            file.write(str(self.likl_func) + '\n' + str(self.likl_grad))

    def load_model(self):
        with open(os.path.join(self.model_dir, 'weights.pkl'), 'rb') as file:
            self.w = pickle.load(file)
        with open(os.path.join(self.model_dir, 'features_id_dict.pkl'), 'rb') as file:
            self.features_id_dict = pickle.load(file)
        with open(os.path.join(self.model_dir, 'bars.pkl'), 'rb') as file:
            self.bars_dict = pickle.load(file)

    def test(self, on='test'):
        self.load_model()

        print(f"Beginning test on {on}...")

        if on == 'test':
            df = pd.read_csv(self.test_path)
        elif on == 'train':
            df = pd.read_csv(self.train_path)
        else:
            raise Exception('invalid test name')
        self.col_map = {name: i for i, name in enumerate(df.columns)}

        with open(os.path.join(self.model_dir, f"results_{on}.txt"), 'w') as file:
            file.write('')
        all_t_tags, all_p_tags = [], []

        grouped = df.groupby(['sequence'])
        print('len =', len(grouped))
        last_group = {'year': None, 'month': None}
        for group_id, (group, df_seq) in enumerate(grouped):
            curr_group = df_seq[['year', 'month']].iloc[-1].to_dict()
            if group_id == 0 or \
                    ('full' not in self.model_name and 'winter' not in self.model_name and
                     last_group['year'] != curr_group['year']) or \
                    ('winter' in self.model_name and curr_group['month'] != '1'):
                df_seq = self.add_stars(df_seq)
            df_seq = df_seq.values.tolist()

            all_t_tags += [row_j[self.col_map['Temp']] for row_j in df_seq[self.markov:]]
            all_p_tags += self.viterbi_beam(df_seq)[self.markov:]
            self.evaluate(all_t_tags, all_p_tags, on)

            last_group = curr_group

        self.evaluate(all_t_tags, all_p_tags, on, do_print=True)

    def evaluate(self, all_t_tags, all_p_tags, on, do_print=False):
        """calculates the accuracy and confusion matrix for prediction"""
        # all_p_tags = [int(item) for sublist in all_p_tags for item in sublist]
        # all_t_tags = [int(item) for sublist in all_t_tags for item in sublist]

        results = 'MSE: ' + str(mean_squared_error(all_t_tags, all_p_tags))
        results += ' accuracy: ' + str(accuracy_score(all_t_tags, all_p_tags))
        if do_print:
            print(results)
        with open(os.path.join(self.model_dir, f"results_{on}.txt"), 'a') as file:
            file.write(results)
            file.write('\n')

    def pi_q_beam(self, pi, history_list, k, t, x):
        """calculate pi"""
        if (k - 1, (t,) + x[:-1]) not in pi:
            return -1

        history_list[0][self.col_map['Temp']] = t
        for i, row in enumerate(history_list[1:]):
            row[self.col_map['Temp']] = x[i]

        feature_dict = self.history2features(history_list, x[-1])
        nome = np.exp(self.w[feature_dict].sum())
        deno = nome
        for y_ in self.tag_set:
            if y_ == x[-1]:
                continue
            feature_dict = self.history2features(history_list, y_)
            deno += np.exp(self.w[feature_dict].sum())
        return pi[(k - 1, (t,) + x[:-1])] * nome / deno

    def viterbi_beam(self, df_seq):
        """perform viterbi"""
        # df_seq = self.add_stars(df_seq).values.tolist()
        pi = {}
        bp = {}
        pi[self.markov - 1, ('*',) * self.markov] = 1
        for k in range(self.markov, len(df_seq)):
            history_list = df_seq[k - self.markov:k + 1]
            b_best_pi = {}
            pi_k = {}
            bp_k = {}

            if k < self.markov * 2:
                ss = [['*']] * (self.markov * 2 - k) + [self.tag_set] * (k - self.markov + 1)
            else:
                ss = [self.tag_set] * (self.markov + 1)

            for x in itertools.product(*ss[1:]):
                bp_k[(k, x)] = max(ss[0], key=lambda t: self.pi_q_beam(pi, history_list, k, t, x))
                pi_k[(k, x)] = pi_calc = self.pi_q_beam(pi, history_list, k, bp_k[(k, x)], x)

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
                     key=lambda x: self.get_pi_beam(pi, len(df_seq) - 1, x)))
        for k in reversed(range(self.markov, len(df_seq))):
            t = [bp[(k, tuple(t[:self.markov]))]] + t
        return t

    @staticmethod
    def get_pi_beam(pi, k, x):
        if (k, x) not in pi:
            return -1
        return pi[(k, x)]


if __name__ == '__main__':
    print('\nWelcome to our accidents predictor!\n')
    # while True:
    #     run_train = input("Should we train before testing?[y/n]   ")
    #     if run_train.lower() in ['y', 'n', 'yes', 'no']:
    #         run_train = True if run_train in ['y', 'yes'] else False
    #         break
    #     print("invalid input. please enter 'y' or 'n'")

    run_train = True
    # run_train = False

    for freq_range in [[str(i) for i in range(1, 13)],  # month
                       ['autumn', 'winter', 'spring', 'monsoon', 'summer'],  # season
                       ['full']]:  # full year
        for freq_part in freq_range:
            for day_part in ['all', 'night', 'morning', 'noon', 'evening']:
                print('\n')
                print('_'.join([freq_part, day_part]))
                model = Memm(model_name='_'.join([freq_part, day_part]))
                if run_train:
                    model.train()
                model.test(on='test')
                model.test(on='train')
