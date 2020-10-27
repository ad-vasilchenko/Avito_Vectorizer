from typing import List


class CountVectorizer ():
    def __init__(self):
        self.features = []

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """'Learn the vocabulary dictionary and return document-term matrix"""
        features = {}

        # creating features dict
        for s in corpus:
            for w in s.lower().split(' '):
                if w not in features:
                    features[w] = len(features)

        # filling count_matrix
        count_matrix = []

        for i, s in enumerate(corpus):
            count_matrix.append([0] * len(features))
            for w in s.lower().split(' '):
                count_matrix[i][features[w]] += 1

        self.features = [x[0] for x in sorted(list(features.items()), key=lambda x: x[1])]

        return count_matrix

    def get_feature_names(self) -> List[str]:
        """Returns list of features names"""
        return(self.features)
