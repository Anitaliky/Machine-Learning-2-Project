import itertools
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from numpy.linalg import norm
import numpy as np
import pickle
import os
import scipy.optimize
import pandas as pd

pd.set_option('display.max_columns', None)


class Memm:

    def __init__(self):
        self.model_dir = '/home/student/Desktop/ML/MEMM/saves/'.format()
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.likl_func = []
        self.likl_grad = []
        self.tag_set = [0, 1]
        self.histories = []
        self.n_total_features = 0  # Total number of features accumulated

        self.cate_map = [('Day_of_Week', '01'), ('Weekend', '02'), ('Month', '03'), ('Year', '04')]

        self.cont_map = [('Temperature(F)', '11'), ('Wind_Chill(F)', '12'), ('Humidity(%)', '13'),
                         ('Pressure(in)', '14'), ('Visibility(mi)', '15'),
                         ('Wind_Speed(mph)', '16'), ('Precipitation(in)', '17')]

        self.features_codes = ['f100'] + ['f' + i + c[1] for c in self.cate_map + self.cont_map for i in ['1', '2']]
        self.features_count = {f_name: {} for f_name in self.features_codes}

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

        self.markov = 3
        self.threshold = {f_name: 0 for f_name in
                          self.features_codes}  # feature count threshold - empirical count must be higher than this
        self.la = 0.1
        self.B = 7
        self.hyper_str = str(self.threshold) + '_' + str(self.la)

    @staticmethod
    def tup_of_tups(df):
        return tuple(df.itertuples(index=False, name=None))

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

    def addstars(self, df_group):
        for i in range(self.markov):
            df_group.loc[-i] = ['*'] * len(df_group.columns)  # adding a row
        df_group.index = df_group.index + self.markov  # shifting index
        df_group = df_group.sort_index().reset_index(drop=True)
        return df_group

    def add2count(self, key, dictionary):
        """"add 1 to the key in the dictionary (or create the key with value 1)."""
        if key not in self.features_count[dictionary]:
            self.features_count[dictionary][key] = 1
        else:
            self.features_count[dictionary][key] += 1

    def add2count_cont(self, df, row_i, curr_dangerous, col_name, code):
        bars = self.create_bars(df, col_name)
        bar_i = self.get_bar(bars, row_i[col_name])
        self.bars_dict[col_name] = bars
        self.add2count((bar_i, curr_dangerous), code)
        return bar_i

    def create_features(self):
        """defines and create the features, counting number of occurence of each one."""
        df = pd.read_csv('/home/student/Desktop/ML/save.csv')
        print('len', len(df))

        grouped = df.groupby(['State', 'Day_Part'])
        print('num of groups', len(grouped), '(shouldnt be more that 156)')
        for group_id, (group, df_group) in enumerate(grouped):
            print(group_id, len(df_group), end=', ')
            df_group = self.addstars(df_group)
            for i in range(self.markov, len(df_group)):  # iterate over indices of words in sentence
                row_i = df_group.iloc[i]
                history_df = df_group.iloc[i - self.markov:i + 1].reset_index(drop=True)
                curr_dangerous = df_group.loc[i, 'Dangerous']
                self.cols_dict = dict(zip(df.columns, range(len(df))))
                self.histories.append((history_df, curr_dangerous))  # add to histories this history and tag

                # categorical
                for col_name, code in self.cate_map:
                    self.add2count((row_i[col_name], curr_dangerous), 'f1' + code)

                bars_i = {}
                # continuous
                for col_name, code in self.cont_map:
                    bars_i[col_name] = self.add2count_cont(df, row_i, curr_dangerous, col_name, 'f1' + code)

                for j in range(self.markov):  # TODO: maybe range(1, self.markov)
                    if history_df.loc[j, 'Dangerous'] == '*':
                        continue
                    self.add2count((history_df.loc[j, 'Dangerous'], j, curr_dangerous), 'f100')

                    # categorical
                    for col_name, code in self.cate_map:
                        self.add2count((history_df.loc[j, col_name], j, curr_dangerous), 'f2' + code)

                    # continuous
                    for col_name, code in self.cont_map:
                        temp_bar_j = self.get_bar(self.bars_dict[col_name], history_df.loc[j, col_name])
                        self.add2count((bars_i[col_name], temp_bar_j, j, curr_dangerous), 'f2' + code)

        print('finished groups')

    def preprocess_features(self):
        """filter features that occured in train set less than threshold,
        and gives an ID for each one."""
        for feature_code, feature_dict in self.features_count.items():
            self.n_dict[feature_code] = self.n_total_features
            for key, count in feature_dict.items():
                if count >= self.threshold[feature_code]:
                    self.features_id_dict[feature_code][key] = self.n_dict[feature_code]
                    self.n_dict[feature_code] += 1
            self.n_total_features = self.n_dict[feature_code]
            print(feature_code, self.n_total_features)
        print(self.features_id_dict)

    def history2features(self, history_df, curr_dangerous):
        """return a list of features ID given a history"""
        features = []
        row_i = history_df.iloc[-1]

        # categorical
        for col_name, code in self.cate_map:
            key = (row_i[col_name], curr_dangerous)
            if key in self.features_id_dict['f1' + code]:
                features.append(self.features_id_dict['f1' + code][key])

        bars_i = {}
        # continuous
        for col_name, code in self.cont_map:
            bars_i[col_name] = self.get_bar(self.bars_dict[col_name], row_i[col_name])
            key = (bars_i[col_name], curr_dangerous)
            if key in self.features_id_dict['f1' + code]:
                features.append(self.features_id_dict['f1' + code][key])

        for j in range(self.markov):
            if history_df.loc[j, 'Dangerous'] == '*':
                continue

            # categorical
            for col_name, code in self.cate_map:
                key = (history_df.loc[j, col_name], j, curr_dangerous)
                if key in self.features_id_dict['f1' + code]:
                    features.append(self.features_id_dict['f1' + code][key])

            # continuous
            for col_name, code in self.cont_map:
                bar_j = self.get_bar(self.bars_dict[col_name], history_df.iloc[j][col_name])
                key = (bars_i[col_name], bar_j, j, curr_dangerous)
                if key in self.features_id_dict['f2' + code]:
                    features.append(self.features_id_dict['f2' + code][key])

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
            self.all_tags_features[(self.tup_of_tups(history), 1 - cur_tag)] = self.history2features(history,
                                                                                                     1 - cur_tag)

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

        print('linear_term', t1 - t0, linear_term)
        print('normalization_term', t2 - t1, normalization_term)
        likelihood = linear_term - normalization_term - 0.5 * self.la * (np.linalg.norm(w) ** 2)
        return -likelihood

    def f_prime(self, w):
        """calculates the likelihood function and its gradient."""
        print('gradient run', w.sum())
        t2 = time.time()
        expected_counts = self.expected_counts(w)
        t3 = time.time()

        print('expected_counts', t3 - t2, expected_counts[-10:])
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

    def train(self):
        """train the model"""
        print('Beginning train...')

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
            print(df_group['Dangerous'])
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
        # with open(self.model_dir + "count_cm.txt", 'w') as file:
        #     file.write(count_cm)

    def pi_q_beam(self, pi, history_df, k, t, x):
        """calculate pi"""
        if (k - 1, (t,) + x[:-1]) not in pi:
            return -1
        history_df['Dangerous'] = [t, ] + list(x)
        feature_dict = self.history2features(history_df, x[-1])
        nome = np.exp(self.w[feature_dict].sum())
        feature_dict = self.history2features(history_df, 1 - x[-1])
        deno = np.exp(self.w[feature_dict].sum()) + nome

        return pi[(k - 1, (t,) + x[:-1])] * nome / deno

    def viterbi_beam(self, df_group):
        """perform viterbi"""
        print('len', len(df_group))
        df_group = self.addstars(df_group)

        pi = {}
        bp = {}
        pi[self.markov - 1, ('*',) * self.markov] = 1
        for k in range(self.markov, len(df_group)):
            print(k, end=', ')
            history_df = df_group.iloc[k - self.markov:k + 1].reset_index(drop=True)
            b_best_pi = {}
            pi_k = {}
            bp_k = {}

            if k < self.markov * 2:
                ss = [['*']] * (self.markov * 2 - k) + [self.tag_set] * (k - self.markov + 1)
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
                     key=lambda x: self.get_pi_beam(pi, len(df_group) - 1, x)))
        for k in reversed(range(self.markov, len(df_group))):
            t = [bp[(k, tuple(t[:self.markov]))]] + t
        print(t)
        return t


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

    model = Memm()
    if run_train:
        model.train()
    model.test(run_train=run_train)
