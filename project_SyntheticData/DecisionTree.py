import pandas as pd

df = pd.read_csv("C:\data\AdultSalary_header.csv")
df = df.iloc[:10, :]

import math
from collections import Counter, defaultdict
from functools import partial
import numpy as np
import scipy.stats


class DecisionTree():
    def __init__(self, df):
        self.df = df
        self.num_attr = len(df.iloc[0, :]) - 1
        self.num_labels = len((Counter(self._labeling(df)).keys()))

    # label 제외하고 dict형태로 변환
    def _get_data(self, df):
        adult = df.iloc[:, :-1]
        df_dict = adult.to_dict('record')
        return df_dict

    # label들을 list로 만듦.

    def _labeling(self, df):
        labels = df.iloc[:, -1]
        return labels.to_list()

    def comb_data(self, df):
        data = self._get_data(df)
        labels = self._labeling(df)
        inputs = []
        for pair in zip(data, labels):
            inputs.append(pair)
        return inputs

        # 클래스에 속할 확률 입력하면 엔트로피 구하는 함수

    def _entropy(self, class_probabilities):
        return sum(-p * math.log(p, 2) for p in class_probabilities if p != 0)

    # 각 레이블의 가중치를 구하는 메소드
    # 해당 레이블 개수/전체 레이블 개수
    def _class_probabilities(self, labels):
        total_count = len(labels)
        return [float(count) / float(total_count) for count in Counter(labels).values()]

    # 노드의 엔트로피 계산하는 메소드
    def _data_entropy(self, labeled_data):
        labels = [label for _, label in labeled_data]
        ''' _ 이게 있어야 True, False만 걸러져서 나옴. 없으면 전체 튜플이 나옴.'''
        # for label in labeled_data:
        #   print('label = {}'.format(label))
        # print('labels = {}'.format(labels))

        '''labels = [False, False, False, True, True] 이렇게 나옴.'''
        probabilities = self._class_probabilities(labels)
        return self._entropy(probabilities)

    # 분할노드의 엔트로피를 정의하는 메소드
    # SUM(자식노드(subset)의 엔트로피 * 가중치)가 분할노드(부모노드)의 엔트로피가 됨.
    def _partition_entropy(self, subsets):
        '''subsets는 inputs 데이터를 하나의 속성으로 구분해놓은 상태.'''
        total_count = sum(len(subset) for subset in subsets)
        ''' subset = [({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False), ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'yes'}, False), ({'level': 'Senior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, False), ({'level': 'Senior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True), ({'level': 'Senior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'yes'}, True)]
            subset = [({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True), ({'level': 'Mid', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, True), ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, True), ({'level': 'Mid', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}, True)]
            subset = [({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True), ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True), ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, False), ({'level': 'Junior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'no'}, True), ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, False)].'''
        return sum(self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets)

    def _partition_by_cat(self, inputs, attribute):
        groups = defaultdict(list)
        # categorical_data = self.df[self.split()[0]].to_dict('record')
        # 한번에 inputs를 받아서 카테고리 컬 데이터만 뽑은 다음에 groups를 만들기 근데 이렇게 하기 위해서는 전처리클래스를 상속받아야만 가능함
        for input in inputs:
            '''
            ({'age': 39, 'workclass': ' State-gov', 'fnlwgt': 77516, 'education': ' Bachelors', 'edunum': 13, 'marital_status': ' Never-married', 'occupation': ' Adm-clerical', 'relationship': ' Not-in-family', 'race': ' White', 'sex': ' Male', 'capital_gain': 2174, 'capital_loss': 0, 'hours_per_works': 40, 'native_country': ' United-States'}, ' <=50K')
            얘가 input '''
            key = input[0][attribute]  # key는 workclass의 value들.
            # print(key)   # 계속 돌다가 salary_class(레이블임) 까지 가면 에러 뜸.
            groups[key].append(input)
        return groups

    def _partition_by_con(self, inputs, attribute):
        Q1 = self.df[attribute].quantile(.25)
        Q2 = self.df[attribute].quantile(.5)
        Q3 = self.df[attribute].quantile(.75)
        groups = defaultdict(list)  # q1 담을 곳
        for input in inputs:
            if input[0][attribute] <= Q1:
                key = '%s_lower_Q1' % (attribute)
                groups[key].append(input)
            elif input[0][attribute] <= Q2:
                key = '%s_between_Q1,Q2' % (attribute)
                groups[key].append(input)
            elif input[0][attribute] <= Q3:
                key = '%s_between_Q2,Q3' % (attribute)
                groups[key].append(input)
            else:
                key = '%s_upper_Q3' % (attribute)
                groups[key].append(input)
        return groups

    def _split(self, df):
        df = df.iloc[:, :-1]
        first = df.iloc[0, :]
        attributes = df.columns
        continuous = []
        categorical = []
        for i in range(len(first) - 1):
            if str(first[i]).isdigit():
                continuous.append(attributes[i])
            else:
                categorical.append(attributes[i])
        return categorical, continuous

    def _get_partitions(self, df, inputs, attribute):
        if attribute in self._split(df)[0]:  # 카테고리컬 데이터인 경우.
            partitions = self._partition_by_cat(inputs, attribute)
        elif attribute in self._split(df)[1]:  # 양적 데이터인 경우.
            partitions = self._partition_by_con(inputs, attribute)
        return partitions

    def _partition_entropy_by(self, df, inputs, attribute):
        return self._partition_entropy(self._get_partitions(df, inputs, attribute).values())

    def build_tree(self, df, inputs, split_candidates=None, max_depth=3):
        if split_candidates is None:
            split_candidates = self._split(df)[0] + self._split(df)[1]

        label_list = [i[1] for i in inputs]
        keys = list(Counter(label_list).keys())
        values = list(Counter(label_list).values())
        half = self.num_labels // 2

        if self.num_attr - len(split_candidates) == max_depth or len(values) <= half:
            return (keys[values.index(max(values))])

        if not split_candidates:
            return (keys[values.index(max(values))])

        best_attribute = min(split_candidates, key=partial(self._partition_entropy_by, df, inputs))

        partitions = self._get_partitions(df, inputs, best_attribute)

        new_candidates = [a for a in split_candidates if a != best_attribute]

        subtrees = {attribute_value: self.build_tree(df, subset, new_candidates) for attribute_value, subset in
                    partitions.items()}

        subtrees[None] = (keys[values.index(max(values))])

        return (best_attribute, subtrees)


class Classify():
    def __init__(self, df, inputs):
        self.df = df
        self.inputs = inputs

    def classify_tree(self, tree, input, inputs):
        label_list = [i[1] for i in inputs]
        keys = list(Counter(label_list).keys())

        if tree in keys:
            return tree

        attribute, subtree_dict = tree

        subtree_key = input.get(attribute)
        if subtree_key not in subtree_dict:
            subtree_key = None

        subtree = subtree_dict[subtree_key]
        return self.classify_tree(subtree, input, inputs)


class SynData(DecisionTree, Classify):
    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        self.df = df

    def _get_value(self, df, label, j):
        inputs_df = df.drop([df.index[j]])
        inputs = self.comb_data(inputs_df)
        input_df = df.iloc[j:j + 1, :-1]
        input = input_df.to_dict('record')[0]
        tree = self.build_tree(inputs_df, inputs)
        value = self.classify_tree(tree, input, inputs)
        return value

    def get_syndata(self, df):
        syn_df = pd.DataFrame(index=range(0, df.shape[0]), columns=df.columns.values.tolist())  # df와 크키 같은 빈 df 생성

        for i in range(df.shape[1]):
            label = df.columns.values.tolist()[-1]

            df_ = df.copy()
            if str(df_[label][0]).isdigit():
                df_["level"] = ""
                df_ = df_.sort_values(by=label)
                for i in df_[label].to_list():
                    if i <= df_[label].quantile(q=0.1):
                        for j in df_[df_[label] == i].index:
                            df_.loc[j, 'level'] = "0.1"
                    elif i <= df_[label].quantile(q=0.2):
                        for j in df_[df_[label] == i].index:
                            df_.loc[j, 'level'] = "0.2"
                    elif i <= df_[label].quantile(q=0.3):
                        for j in df_[df_[label] == i].index:
                            df_.loc[j, 'level'] = "0.3"
                    elif i <= df_[label].quantile(q=0.4):
                        for j in df_[df_[label] == i].index:
                            df_.loc[j, 'level'] = "0.4"
                    elif i <= df_[label].quantile(q=0.5):
                        for j in df_[df_[label] == i].index:
                            df_.loc[j, 'level'] = "0.5"
                    elif i <= df_[label].quantile(q=0.6):
                        for j in df_[df_[label] == i].index:
                            df_.loc[j, 'level'] = "0.6"
                    elif i <= df_[label].quantile(q=0.7):
                        for j in df_[df_[label] == i].index:
                            df_.loc[j, 'level'] = "0.7"
                    elif i <= df_[label].quantile(q=0.8):
                        for j in df_[df_[label] == i].index:
                            df_.loc[j, 'level'] = "0.8"
                    elif i <= df_[label].quantile(q=0.9):
                        for j in df_[df_[label] == i].index:
                            df_.loc[j, 'level'] = "0.9"
                    else:
                        for j in df_[df_[label] == i].index:
                            df_.loc[j, 'level'] = "1.0"

                max = {i: df_[df_['level'] == i][label].max() for i in list(df_['level'].unique())}
                min = {i: df_[df_['level'] == i][label].min() for i in list(df_['level'].unique())}
                std = {i: np.std(df_[df_['level'] == i][label]) for i in list(
                    df_['level'].unique())}
                mean = {i: df_[df_['level'] == i][label].mean() for i in list(df_['level'].unique())}
                dtype = df_[label].dtype
                df_ = df_.drop([label], axis=1)  # 이걸 굳이 할 필요가 있나!?!? 어차피 뒤에 df_ 쓰는게 없는거 같아서!

                for j in range(df_.shape[0]):
                    value = self._get_value(df_, label, j)
                    syn_df.loc[j, label] = value

                syn_df = syn_df.sort_values(by=label)
                syn_level = list(syn_df[label].unique())
                freq = {i: syn_df.groupby(label).size()[i] for i in syn_level}

                group = defaultdict(list, {i: [min[i], max[i], mean[i], std[i], freq[i]] for i in syn_level})

                np.random.seed(0)  # 넣어야 하나 말아야 하나 ?_?
                rand = defaultdict(list, {k: [] for k in syn_level})  # 난수 담을거
                for i, j in group.items():
                    while len(rand[i]) < j[4]:
                        x = np.random.normal(j[2], j[3])
                        if j[0] <= x <= j[1]:
                            rand[i].append(x)
                rand_list = [np.sort(i) for i in rand.values()]

                if dtype == 'int64':  # 만약 데이터타입이 정수이면
                    rand_int = [list(map(int, i)) for i in rand_list]
                    syn_df[label] = np.concatenate(rand_int)
                else:
                    syn_df[label] = np.concatenate(rand_list)

            else:
                for j in range(df.shape[0]):
                    value = self._get_value(df, label, j)
                    syn_df.loc[j, label] = value

            # label열 맨 앞으로 옮기기
            new_columns = df.columns.values.tolist()[:-1]
            new_columns.insert(0, label)
            df = df[new_columns]
        return syn_df


# print(df, '\n')

syn_data = SynData(df)
result = syn_data.get_syndata(df)
print(result)
result.to_csv('syn_data.csv')