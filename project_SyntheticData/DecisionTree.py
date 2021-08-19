import pandas as pd

df = pd.read_csv("C:\data\AdultSalary_header.csv")
df = df.iloc[:200, :]


import math
from collections import Counter, defaultdict
from functools import partial


class DecisionTree():
    def __init__(self, df):
        self.df = df
        self.num_attr = len(df.iloc[0, :])-1
        self.num_labels = len((Counter(self._labeling()).keys()))

    # label 제외하고 dict형태로 변환
    def _get_data(self):
        adult = self.df.iloc[:, :-1]
        df_dict = adult.to_dict('record')
        return df_dict

    # label들을 list로 만듦.

    def _labeling(self):
        labels = self.df.iloc[:, -1]
        return labels.to_list()

    def comb_data(self):
        data = self._get_data()
        labels = self._labeling()
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
        ''' Counter(labels) = {0 : 3, 1 : 2} 이런식으로 나옴. 
        return값은 0.6, 0.4 이런식으로 나옴.'''

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
        groups = defaultdict(list) # q1 담을 곳
        for input in inputs:
            if input[0][attribute] <= Q1:
                key = '%s_lower_Q1' %(attribute)
                groups[key].append(input)
            elif input[0][attribute] <= Q2:
                key = '%s_between_Q1,Q2' %(attribute)
                groups[key].append(input)
            elif input[0][attribute] <= Q3:
                key = '%s_between_Q2,Q3' %(attribute)
                groups[key].append(input)
            else:
                key = '%s_upper_Q3' %(attribute)
                groups[key].append(input)
        return groups

    def _split(self):
        df = self.df.iloc[:, :-1]
        first = self.df.iloc[0, :]
        attributes = self.df.columns
        continuous = []
        categorical = []
        for i in range(
                len(first) - 1):  # 마지막 column 을 빼기위해 range() 에서 -1해줌. 원래는 lable column을 빼줘야함. --> 나중에 재현데이터 할 때 손봐야함.
            if str(first[i]).isdigit():
                continuous.append(attributes[i])
            else:
                categorical.append(attributes[i])
        return categorical, continuous

    def _get_partitions(self, inputs, attribute):
        if attribute in self._split()[0]:  # 카테고리컬 데이터인 경우.
            partitions = self._partition_by_cat(inputs, attribute)
        elif attribute in self._split()[1]:  # 양적 데이터인 경우.
            partitions = self._partition_by_con(inputs, attribute)
        return partitions

    def _partition_entropy_by(self, inputs, attribute):
        return self._partition_entropy(self._get_partitions(inputs, attribute).values())

    def build_tree(self, inputs, split_candidates=None, max_depth=3):
        if split_candidates is None:
            split_candidates = self._split()[0] + self._split()[1]

        label_list = [i[1] for i in inputs]
        keys = list(Counter(label_list).keys())
        values = list(Counter(label_list).values())
        half = self.num_labels // 2

        if self.num_attr-len(split_candidates) == max_depth or len(values) <= half:
            return (keys[values.index(max(values))])

        if not split_candidates:
            return (keys[values.index(max(values))])

        best_attribute = min(split_candidates, key=partial(self._partition_entropy_by, inputs))

        partitions = self._get_partitions(inputs, best_attribute)

        new_candidates = [a for a in split_candidates if a != best_attribute]


        subtrees = {attribute_value: self.build_tree(subset, new_candidates) for attribute_value, subset in partitions.items()}

        subtrees[None] = (keys[values.index(max(values))])

        return (best_attribute, subtrees)


class Classify():
    def __init__(self, df, inputs):
        self.df = df
        self.inputs = inputs

    def classify_tree(self, tree, input):
        label_list = [i[1] for i in inputs]
        keys = list(Counter(label_list).keys())   # 전체 레이블 종류를 나타내는 list

        if tree in keys:  # 마지막에 leaf node 결정할 때 사용되는 코드
            return tree

        attribute, subtree_dict = tree  # attribute: 분기한 속성, subtree_dict: 분기 결과

        subtree_key = input.get(attribute)  # dictionary함수에서 attribute에 대응하는 value값을 받아옴.
        if subtree_key not in subtree_dict:
            subtree_key = None

        subtree = subtree_dict[subtree_key]
        return self.classify_tree(subtree, input)


#얘네는 NewDF class에서 갖고 들어올 값(input)
b = df.iloc[197:198, :-1]
input = b.to_dict('record')[0]

dtree = DecisionTree(df)
inputs = dtree.comb_data()
tree = dtree.build_tree(inputs)
print(tree)

a = Classify(df, inputs)
label = a.classify_tree(tree, input)
print(label)





