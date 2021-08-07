import pandas as pd

df = pd.read_csv("C:\data\AdultSalary_header.csv")
df = df.iloc[:100, :]


import math
from collections import Counter, defaultdict
from functools import partial


class DecisionTree():
    def __init__(self, df):
        self.df = df

    # label 제외하고 dict형태로 변환
    def _get_data(self):
        adult = self.df.iloc[:, :-1]
        df_dict = adult.to_dict('record')
        return df_dict

    # label들을 list로 만듦.
    def _labeling(self):
        labels = self.df.iloc[:, -1]
        return labels.to_list()

    # 위의 두 return 값들을 zip함수 사용하여 원하는 데이터 형태인 inputs을 return
    # 이제 Decisino Tree class에 적용할 수 있는 데이터 형태 만들어짐.
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

    def _partition_entropy_by(self, inputs, attribute):
        if attribute in self._split()[0]:  # 카테고리컬 데이터인 경우.
            partitions = self._partition_by_cat(inputs, attribute)
        elif attribute in self._split()[1]:  # 양적 데이터인 경우.
            partitions = self._partition_by_con(inputs, attribute)
        return self._partition_entropy(partitions.values())


    def build_tree(self, inputs, split_candidates=None):
        if split_candidates is None:
            split_candidates_cat = self._split()[0]  # 여기서 먼저 양적, 카테고리컬 속성 후보 나눔.
            split_candidates_con = self._split()[1]
            split_candidates = split_candidates_cat + split_candidates_con
            # print(split_candidates)
            # print(split_candidates_cat, split_candidates_con)

        # 얜 지금 이분법 categorical 문제.
        num_inputs = len(inputs)
        num_class0 = len([label for _, label in inputs if label == ' <=50K'])
        num_class1 = num_inputs - num_class0

        if num_class0 == 0:
            return ' >50K'
        if num_class1 == 0:
            return ' <=50K'

        if not split_candidates:
            if num_class0 >= num_class1:
                return ' <=50K'
            else:
                return ' >50K'

        best_attribute = min(split_candidates, key=partial(self._partition_entropy_by, inputs))
        # print(best_attribute)

        if best_attribute in self._split()[0]:  # 카테고리컬일때
            partitions = self._partition_by_cat(inputs, best_attribute)

        elif best_attribute in self._split()[1]:  # 양적일때
            partitions = self._partition_by_con(inputs, best_attribute)

        new_candidates = [a for a in split_candidates if a != best_attribute]
        # print(new_candidates)

        subtrees = {attribute_value: self.build_tree(subset, new_candidates) for attribute_value, subset in partitions.items()}


        if num_class0 >= num_class1:
            subtrees[None] = ' <=50K'
        elif num_class0 < num_class1:
            subtrees[None] = ' >50K'
        return (best_attribute, subtrees)


class Classify:

    # def __init__(self, tree, input):
    #     self.tree = tree
    #     self.input = input

    def classify_tree(self, tree, input):
        if tree in [' <=50K', ' >50K']:  # 마지막에 leaf node 결정할 때 사용되는 코드
            return tree

        attribute, subtree_dict = tree  # 위에서 return 받은 (best_attribute, subtrees) 를 받는 것.

        subtree_key = input.get(attribute)  # dictionary함수에서 key에 대응하는 value값을 받아옴.
        if subtree_key not in subtree_dict:
            subtree_key = None
        # print(subtree_key, end='\n')

        subtree = subtree_dict[subtree_key]
        return self.classify_tree(subtree, input)


test_dt = DecisionTree(df)
inputs = test_dt.comb_data()
tree = test_dt.build_tree(inputs)
print(tree)


