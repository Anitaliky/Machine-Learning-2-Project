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
        self.features_list = ['f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f108', 'f109', 'f110']
        self.features_list = ['f100', 'f101', 'f102', 'f104', 'f110']
        self.features_count = {f_name: {} for f_name in self.features_list}
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this
        self.la = la
        self.B = 7
        self.hyper_str = str(self.threshold) + '_' + str(self.la)
        self.model_dir = 'saves/'.format()
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

    def tup_of_tups(self, df):
        return tuple(df.itertuples(index=False, name=None))

    def add2dict(self, key, dictionary):
        """"add 1 to the key in the dictionary (or create the key with value 1)."""
        if key not in self.features_count[dictionary]:
            self.features_count[dictionary][key] = 1
        else:
            self.features_count[dictionary][key] += 1

    def create_features(self):
        """defines and create the features, counting number of occurence of each one."""
        df = pd.read_csv('/home/student/Desktop/ML/dfs.csv').head(10000)
        # df = df[df['Airport_Code']=='KCMH']
        df['Severity'] = df['Severity'].replace({1: 0})
        df['Severity'] = df['Severity'].replace({2: 0})
        df['Severity'] = df['Severity'].replace({3: 1})
        df['Severity'] = df['Severity'].replace({4: 1})

        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        df['Temperature(F)'] = min_max_scaler.fit_transform(df['Temperature(F)'].values.reshape(-1, 1))
        df['Distance(mi)'] = min_max_scaler.fit_transform(df['Distance(mi)'].values.reshape(-1, 1))
        df['Start_Lat'] = min_max_scaler.fit_transform(df['Start_Lat'].values.reshape(-1, 1))
        df['Start_Lng'] = min_max_scaler.fit_transform(df['Start_Lng'].values.reshape(-1, 1))

        grouped = df.groupby('Airport_Code')
        print('num of groups', len(grouped))
        for group_id, (group, df_group) in enumerate(grouped):
            print(group_id, len(df_group), end=', ')
            for i in range(10):
                df_group.loc[-i] = ['*'] * len(df_group.columns)  # adding a row
            df_group.index = df_group.index + 10  # shifting index
            df_group = df_group.sort_index().reset_index(drop=True)
            # print('df_group', df_group.head(12))
            for i in range(10, len(df_group)):  # iterate over indices of words in sentence
                history_df = df_group.iloc[i - 10:i + 1].reset_index(drop=True)
                curr_severity = df_group.loc[i, 'Severity']
                self.histories.append((history_df, curr_severity))  # add to histories this history and tag
                # print(i, end=',')
                for j in range(10):  # TODO: maybe range(1, 11)
                    self.add2dict((history_df.loc[j, 'Severity'], j, curr_severity), 'f100')
                    self.add2dict((j, curr_severity), 'f101')  # value will be distance of temperatures.
                    self.add2dict((history_df.loc[j, 'Traffic_Signal'], j, curr_severity),
                                  'f102')  # value will be distance of traffic_signal.
                    # self.add2dict((j, curr_severity), 'f103')  # value will be distance of pressure.
                    self.add2dict((j, curr_severity), 'f104')  # value will be distance of distance.

                for col_name in history_df.columns:
                    if col_name in self.skip_cols:
                        continue
                    self.add2dict((col_name, curr_severity), 'f110')  # value will be the attributes of the row itself.
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

    # history = (sentence_{1:n}(list), T_2, T_1, i)
    def history2features(self, history_df, curr_severity):
        """return a list of features ID given a history"""
        features = {}
        # adds the id of a feature if its conditions happens on the given history and tag
        for j in range(10):
            key = (history_df.loc[j, 'Severity'], j, curr_severity)
            if key in self.features_id_dict['f100']:
                features[self.features_id_dict['f100'][key]] = 1

            key = (j, curr_severity)
            if key in self.features_id_dict['f101']:
                x_j = history_df.loc[j, 'Temperature(F)']
                if x_j != '*':
                    features[self.features_id_dict['f101'][key]] = \
                        abs(history_df.iloc[-1, history_df.columns.get_loc('Temperature(F)')] - x_j)
                    # TODO: try 1 -

            key = (history_df.loc[j, 'Traffic_Signal'], j, curr_severity)
            if key in self.features_id_dict['f102']:
                features[self.features_id_dict['f102'][key]] = 1

            # key = (j, curr_severity)
            # if key in self.features_id_dict['f103']:
            #     x_j = history_df.loc[j, 'Pressure(in)']
            #     if x_j != '*':
            #         features[self.features_id_dict['f103'][key]] = \
            #         abs(history_df.iloc[-1, history_df.columns.get_loc('Pressure(in)')]-x_j)

            key = (j, curr_severity)
            if key in self.features_id_dict['f104']:
                x_j = history_df.loc[j, 'Distance(mi)']
                if x_j != '*':
                    features[self.features_id_dict['f104'][key]] = \
                        abs(history_df.iloc[-1, history_df.columns.get_loc('Distance(mi)')] - x_j)

        for col_name in history_df.columns:
            key = (col_name, curr_severity)
            if key in self.features_id_dict['f110']:
                x_j = history_df.iloc[-1, history_df.columns.get_loc(col_name)]
                if x_j > 1 or x_j < -1:
                    continue
                if x_j != '*':
                    features[self.features_id_dict['f110'][key]] = x_j

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
            linear_sum += (w[list(feature_dict.keys())] * list(feature_dict.values())).sum()
        return linear_sum

    def normalization_term(self, w):
        all_sum = 0
        for history, _ in self.histories:
            log_sum = 0
            for y_ in self.tag_set:
                feature_dict = self.all_tags_features[(self.tup_of_tups(history), y_)]
                log_sum += np.exp((w[list(feature_dict.keys())] * list(feature_dict.values())).sum())
            self.all_tags_exp[self.tup_of_tups(history)] = log_sum
            all_sum += np.log(log_sum)
        return all_sum

    def empirical_counts(self):
        self.train_vector_sum = np.zeros(self.n_total_features)
        for feature_dict in self.train_features.values():
            self.train_vector_sum[list(feature_dict.keys())] += list(feature_dict.values())

    def expected_counts(self, w):
        """calculate the expected term of the loss function"""
        all_sum = np.zeros(self.n_total_features)
        for history, cur_tag in self.histories:
            deno = self.all_tags_exp[self.tup_of_tups(history)]
            # print('deno', deno)
            # time.sleep(3)
            for y_ in self.tag_set:
                feature_dict = self.all_tags_features[(self.tup_of_tups(history), y_)]
                nom = np.exp((w[list(feature_dict.keys())] * list(feature_dict.values())).sum())
                all_sum[list(self.all_tags_features[(self.tup_of_tups(history), y_)].keys())] += (nom / deno)
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
        test_path = 'data/dfs.csv'
        print("Beginning test on", test_path)

        if not run_train:
            with open(self.model_dir + 'weights.pkl', 'rb') as file:
                self.w = pickle.load(file)
            with open(self.model_dir + 'features_id_dict.pkl', 'rb') as file:
                self.features_id_dict = pickle.load(file)

        all_sentences = []  # list of sentences for saving predicted tags in competition
        all_t_tags = []  # list of true tags for comparing on test
        all_p_tags = []  # list of predicted tags
        df = pd.read_csv('/home/student/Desktop/ML/dfs.csv').head(10000)
        grouped = df.groupby('Airport_Code')
        for group_id, (group, df_group) in enumerate(grouped):
            print(group_id, len(df_group), end=', ')
            # if len(df_group) < 20:
            #     continue
            all_t_tags.append(df_group)
            p_tags = self.viterbi(df_group)
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
        nome = np.exp((self.w[list(feature_dict.keys())] * list(feature_dict.values())).sum())
        feature_dict = self.history2features(history_df, 1 - x[-1])
        deno = np.exp(self.w[list(feature_dict.keys())] * list(feature_dict.values())).sum() + nome
        return pi[(k - 1, (t, ) + x[:-1])] * nome / deno

    def viterbi_beam(self, sentence):
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

    def viterbi(self, df_group):
        """perform viterbi"""
        print('len', len(df_group))
        for i in range(10):
            df_group.loc[-i] = ['*'] * len(df_group.columns)  # adding a row
        df_group.index = df_group.index + 10  # shifting index
        df_group = df_group.sort_index().reset_index(drop=True)

        pi = {}
        bp = {}
        # pi = {(k, ('*',) * 10): 1 for }
        pi[9, ('*',) * 10] = 1
        for k in range(10, len(df_group)):
            history_df = df_group.iloc[k - 10:k + 1].reset_index(drop=True)
            curr_severity = df_group.loc[k, 'Severity']

            if k < 20:
                ss = [['*']] * (20 - k) + [self.tag_set] * (k - 9)
            else:
                ss = [self.tag_set] * 11

            for x in itertools.product(*ss[1:]):
                bp[(k, x)] = max(ss[0], key=lambda t: self.pi_q(pi, history_df, k, t, x))
                pi[(k, x)] = self.pi_q(pi, history_df, k, bp[(k, x)], x)
        # print(pi)
        t = list(max([x for x in itertools.product(*ss[1:])],
                 key=lambda x: pi[(len(df_group) - 1, x)]))
        for k in reversed(range(10, len(df_group))):
            t = [bp[(k, tuple(t[:10]))]] + t
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
    thre = 1
    la = 0.1

    model = Mmem(la=la, threshold=thre)
    if run_train:
        model.train()
    model.test(run_train=run_train)
