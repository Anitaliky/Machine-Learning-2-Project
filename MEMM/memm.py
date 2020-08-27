import time
from collections import OrderedDict
from itertools import combinations
from numpy.linalg import norm
import numpy as np
import pickle
import os
import scipy.optimize
import pandas as pd


class Mmem:

    def __init__(self, la, threshold, comp=False, k=None):
        self.likl_func = []
        self.likl_grad = []
        self.tag_set = {0, 1}
        self.histories = []
        self.n_total_features = 0  # Total number of features accumulated
        self.features_list = ['f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f108', 'f109', 'f110']
        self.features_count = {f_name: OrderedDict() for f_name in self.features_list}
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this
        self.la = la
        self.B = 7
        self.hyper_str = str(self.threshold) + '_' + str(self.la)
        self.comp = comp
        self.model_dir = 'saves/'.format()
        if not os.path.exists(self.model_dir) and not self.comp:
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

    def add2dict(self, key, dictionary):
        """"add 1 to the key in the dictionary (or create the key with value 1)."""
        if key not in self.features_count[dictionary]:
            self.features_count[dictionary][key] = 1
        else:
            self.features_count[dictionary][key] += 1

    def create_features(self):
        """defines and create the features, counting number of occurence of each one."""
        df = pd.read_csv('/home/student/Desktop/ML/dfs.csv')
        grouped = df.groupby('Airport_Code')
        for group_id, (group, df_group) in enumerate(grouped):
            print(group_id, len(df_group))
            for i in range(10):
                df_group.loc[-i] = ['*'] * len(df_group.columns)  # adding a row
            df_group.index = df_group.index + 10  # shifting index
            df_group = df_group.sort_index()
            df_group = df_group.reset_index(drop=True)
            # print('df_group', df_group.head(12))
            for i in range(10, len(df_group)):  # iterate over indices of words in sentence
                history_df = df_group.iloc[i - 10:i + 1].reset_index(drop=True)
                curr_severity = df_group.loc[i, 'Severity']
                self.histories.append((history_df, curr_severity))  # add to histories this history and tag
                # print(i, end=',')
                for j in range(10):
                    self.add2dict((history_df.loc[j, 'Severity'], j, curr_severity), 'f100')
                    self.add2dict((j, curr_severity), 'f101')  # value will be distance of temperatures.
                    self.add2dict((history_df.loc[j, 'Traffic_Signal'], j, curr_severity), 'f102')  # value will be distance of traffic_signal.
                    self.add2dict((j, curr_severity), 'f103')  # value will be distance of pressure.
                    self.add2dict((j, curr_severity), 'f104')  # value will be distance of distance.

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

    @staticmethod
    def get_distance(history_df, j, col_name):
        return norm(history_df.loc[j, col_name] - history_df.tail(1)[col_name], 1)
    # TODO: try 1 -

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
                features[self.features_id_dict['f101'][key]] = self.get_distance(history_df, j, 'Temperature(F)')

            key = (history_df.loc[j, 'Traffic_Signal'], j, curr_severity)
            if key in self.features_id_dict['f102']:
                features[self.features_id_dict['f102'][key]] = 1

            key = (j, curr_severity)
            if key in self.features_id_dict['f103']:
                features[self.features_id_dict['f103'][key]] = self.get_distance(history_df, j, 'Pressure(in)')

            key = (j, curr_severity)
            if key in self.features_id_dict['f104']:
                features[self.features_id_dict['f104'][key]] = self.get_distance(history_df, j, 'Distance(mi)')



        # features += [self.features_id_dict['f100'][(cur_word.lower(), cur_tag)]] \
        #     if (cur_word.lower(), cur_tag) in self.features_id_dict['f100'] else []
        # for suffix_len in range(1, 5):  # [1, 2, 3, 4]
        #     features += [self.features_id_dict['f101'][(cur_word.lower()[-suffix_len:], cur_tag)]] \
        #         if (cur_word.lower()[-suffix_len:], cur_tag) in self.features_id_dict['f101'] else []
        # for prefix_len in range(1, 5):  # [1, 2, 3, 4]
        #     features += [self.features_id_dict['f102'][(cur_word.lower()[:prefix_len], cur_tag)]] \
        #         if (cur_word.lower()[:prefix_len], cur_tag) in self.features_id_dict['f102'] else []
        # features += [self.features_id_dict['f103'][(pre_prev_tag, prev_tag, cur_tag)]] \
        #     if (pre_prev_tag, prev_tag, cur_tag) in self.features_id_dict['f103'] else []
        # features += [self.features_id_dict['f104'][(prev_tag, cur_tag)]] \
        #     if (prev_tag, cur_tag) in self.features_id_dict['f104'] else []
        # features += [self.features_id_dict['f105'][(cur_tag,)]] \
        #     if (cur_tag,) in self.features_id_dict['f105'] else []
        # features += [self.features_id_dict['f106'][(prev_word.lower(), cur_tag)]] \
        #     if (prev_word.lower(), cur_tag) in self.features_id_dict['f106'] else []
        # features += [self.features_id_dict['f107'][(next_word, cur_tag)]] \
        #     if (next_word, cur_tag) in self.features_id_dict['f107'] else []
        # features += [self.features_id_dict['f108'][(pre_prev_word.lower(), prev_word.lower(), cur_tag)]] \
        #     if (pre_prev_word.lower(), prev_word.lower(), cur_tag) in self.features_id_dict['f108'] else []
        # features += [self.features_id_dict['f109'][(i == 0, cur_word[0].isupper(), cur_tag)]] \
        #     if (i == 0, cur_word[0].isupper(), cur_tag) in self.features_id_dict['f109'] else []
        # features += [self.features_id_dict['f110'][(cur_word_digit, cur_tag)]] \
        #     if (cur_word_digit, cur_tag) in self.features_id_dict['f110'] else []

        return features


    def extract_features(self):
        for history, cur_tag in self.histories:
            self.train_features[(history, cur_tag)] = self.history2features(history, cur_tag)
            for y_ in self.tag_set:
                self.all_tags_features[(history, y_)] = self.history2features(history, y_)


    def linear_term(self, w):
        """calculate the linear term of the likelihood function: sum for each i: w*f(x_i,y_i)"""
        linear_sum = 0
        for feature_dict in self.train_features.values():
            linear_sum += (w[list(feature_dict.keys())]*list(feature_dict.values())).sum()
        return linear_sum

    def normalization_term(self, w):
        all_sum = 0
        for history, _ in self.histories:
            log_sum = 0
            for y_ in self.tag_set:
                feature_dict = self.all_tags_features[(history, y_)]
                log_sum += np.exp((w[list(feature_dict.keys())] * list(feature_dict.values())).sum())
            self.all_tags_exp[history] = log_sum
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
            deno = self.all_tags_exp[history]
            for y_ in self.tag_set:
                feature_dict = self.all_tags_features[(history, y_)]
                nom = np.exp((w[list(feature_dict.keys())] * list(feature_dict.values())).sum())
                all_sum[list(self.all_tags_features[(history, y_)].values())] += (nom / deno)
        return all_sum

    def target_funcs(self, w):
        """calculates the likelihood function and its gradient."""
        linear_term = self.linear_term(w)
        normalization_term = self.normalization_term(w)
        expected_counts = self.expected_counts(w)

        likelihood = linear_term - normalization_term - 0.5 * self.la * (np.linalg.norm(w) ** 2)
        grad = self.train_vector_sum - expected_counts - self.la * w
        return -likelihood, -grad

    def minimize(self):
        """applies the minimization for the loss function"""
        self.empirical_counts()
        w = np.random.rand(self.n_total_features) / 100 - 0.005  # initiates the weigths to small values
        w, _, _ = scipy.optimize.fmin_l_bfgs_b(func=self.target_funcs, x0=w, maxiter=100,
                                               epsilon=10 ** (-6), iprint=10)
        self.w = w

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
        self.preprocess_features()
        self.extract_features()
        self.minimize()
        self.save_model()

    def test(self, m=None, test_path=None, run_train=True):
        """test the model"""
        if test_path is None:
            test_path = 'data/test.csv'
        print("Beginning test on", test_path)

        if not run_train:
            with open(self.model_dir + 'weights.pkl', 'rb') as file:
                self.w = pickle.load(file)
            with open(self.model_dir + 'features_id_dict.pkl', 'rb') as file:
                self.features_id_dict = pickle.load(file)

        all_sentences = []  # list of sentences for saving predicted tags in competition
        all_t_tags = []  # list of true tags for comparing on test
        all_p_tags = []  # list of predicted tags
        with open(test_path) as file:
            lines = list(file)
        for i, line in enumerate(lines):
            print(i + 1, end=', ')
            sentence = line.split()
            if not self.comp:
                sentence, t_tags = self.separate(sentence)
                all_t_tags.append(t_tags)
            else:
                all_sentences.append(sentence)
            p_tags = self.viterbi_beam(sentence)
            all_p_tags.append(p_tags)
        if not self.comp:
            self.evaluate(all_p_tags, all_t_tags)
        else:
            self.save_tags(all_sentences, all_p_tags)

    def save_tags(self, all_sentences, all_p_tags):
        """when running cometition, save the predicted tags """
        string = ''
        for sentence, tags in zip(all_sentences, all_p_tags):
            for word, tag in zip(sentence, tags):
                string += word + '_' + tag + ' '
            string += '\n'
        with open('../comp_m{}_314828732.wtag'.format(self.name), 'w') as file:
            file.write(string)

    def evaluate(self, all_p_tags, all_t_tags):
        """calculates the accuracy and confusion matrix for prediction"""
        tags = np.concatenate(all_p_tags)
        t_tags = np.concatenate(all_t_tags)
        self.accuracy = np.array(tags == t_tags).mean()

        count_dict = {y_true: {y_pred: 0 for y_pred in self.tag_set} for y_true in self.tag_set}
        for t_tag, p_tag in zip(t_tags, tags):
            try:
                count_dict[t_tag][p_tag] += 1
            except KeyError as e:
                continue
        acc_dict = {}
        for tag in self.tag_set:
            if sum(count_dict[tag].values()) != 0:
                acc_dict[tag] = count_dict[tag][tag] / sum(count_dict[tag].values())
            else:
                acc_dict[tag] = 999  # to prevent seeing tags that didn't appear.
        cols = sorted(acc_dict, key=acc_dict.get)[:10]  # 10 worse tags.
        self.tag_set = cols + list(self.tag_set - set(cols))  # so tag set will have same order as cols.

        cm_table = np.zeros((len(self.tag_set), len(cols)))
        count_table = np.zeros((len(self.tag_set), len(cols)))
        for i, y_pred in enumerate(self.tag_set):
            for j, y_true in enumerate(cols):
                if sum(count_dict[y_true].values()) != 0 and count_dict[y_true][y_pred] != 0:
                    cm_table[i][j] = count_dict[y_true][y_pred] / sum(count_dict[y_true].values())
                    count_table[i][j] = count_dict[y_true][y_pred]
                else:
                    cm_table[i][j] = np.nan  # looks better.
                    count_table[i][j] = np.nan  # looks better.
        percent_cm = self.get_cm(cm_table, cols, self.tag_set) + '\n'
        count_cm = self.get_cm(count_table, cols, self.tag_set) + '\n'
        print("The accuracy is:", self.accuracy)
        # saves results of testing.
        with open(self.model_dir + "accuracy.txt", 'w') as file:
            file.write(str(self.accuracy))
        with open(self.model_dir + "percent_cm.txt", 'w') as file:
            file.write(percent_cm)
        with open(self.model_dir + "count_cm.txt", 'w') as file:
            file.write(count_cm)

    @staticmethod
    def get_cm(table, cols_title, rows_title):
        """generates string of confusion matrix"""
        size = 6
        sep = "  "
        table_str = sep.join(str(x).rjust(size) for x in [""] + cols_title) + '\n'
        for i in range(len(table)):
            row = [round(x, 4) for x in table[i]]
            table_str += sep.join(str(x).rjust(size) for x in [list(rows_title)[i]] + row) + '\n'
        return table_str

    @staticmethod
    def separate(sentence):
        """separates words and tags in a sentence"""
        words, tags = [], []
        for i in range(len(sentence)):
            cur_word, cur_tag = sentence[i].split('_')
            words.append(cur_word)
            tags.append(cur_tag)
        return words, tags

    def pi_q(self, pi, sentence, k, t, u, v):
        """calculate pi"""
        deno = 0
        history = (sentence, t, u, k - 1)
        nome = np.exp(self.w[self.history2features(history, v)].sum())
        for y_ in self.tag_set:
            deno += np.exp(self.w[self.history2features(history, y_)].sum())
        return pi[(k - 1, t, u)] * nome / deno

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

    def viterbi(self, sentence):
        """perform viterbi"""
        pi = {}
        bp = {}
        pi[(0, '*', '*')] = 1
        for k in range(1, len(sentence) + 1):
            if k == 1:
                s_1 = ['*']
                s_2 = ['*']
            elif k == 2:
                s_1 = self.tag_set
                s_2 = ['*']
            else:
                s_1 = self.tag_set
                s_2 = self.tag_set
            for u in s_1:
                for v in self.tag_set:
                    bp[(k, u, v)] = max(s_2, key=lambda t: self.pi_q(pi, sentence, k, t, u, v))
                    pi[(k, u, v)] = self.pi_q(pi, sentence, k, bp[(k, u, v)], u, v)

        t = list(
            max([(u, v) for u in self.tag_set for v in self.tag_set], key=lambda uv: pi[(len(sentence), uv[0], uv[1])]))
        for k in reversed(range(1, len(sentence) - 1)):
            t = [bp[(k + 2, t[0], t[1])]] + t
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
