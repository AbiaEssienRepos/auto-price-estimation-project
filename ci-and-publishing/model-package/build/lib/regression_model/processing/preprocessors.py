import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler


class CategoricalEncoder:
    """Performs one hot encoding on categorical variables"""

    def __init__(self, variables, target):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.target_ = target
        self.labels_ = {}

    def fit(self, X, y):
        # persist encoding mapping to a dictionary

        tmp = pd.concat([X, y], axis=1)
        ranked = rankdata(tmp[self.target_])
        tmp["rank"] = ranked

        for feature in self.variables:
            ordered_labels = tmp.groupby([feature])["rank"].sum().sort_values().index
            ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
            self.labels_[feature] = ordinal_label

        return self

    def transform(self, X, y=None):

        for feature in self.variables:
            X[feature] = X[feature].map(self.labels_[feature])
        return X


class CarTransform:
    def __init__(self, variable):
        self.variable = variable

    def fit(self, X, y=None):
        return self

    def assign_country(self, row):

        # list of original countries of the auto brands
        italy = ["alfa-romeo"]
        germany = ["audi", "bmw", "porsche", "volkswagen"]
        usa = ["chevrolet", "dodge", "buick", "mercury", "plymouth"]
        japan = ["honda", "isuzu", "mazda", "mitsubishi", "nissan", "subaru", "toyota"]
        uk = ["jaguar"]
        france = ["peugeot", "renault"]
        sweden = ["saab", "volvo"]

        if row in italy:
            return "Italy"
        elif row in germany:
            return "Germany"
        elif row in usa:
            return "USA"
        elif row in japan:
            return "Japan"
        elif row in uk:
            return "United Kingdom"
        elif row in france:
            return "France"
        elif row in sweden:
            return "Sweden"

    def transform(self, X, y=None):
        X[self.variable] = X[self.variable].apply(self.assign_country)
        return X


class RemapVariable:
    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.ordinal_labels = {}

    def fit(self, X, y=None):

        for variable in self.variables:
            ordered_labels = X[variable].value_counts().to_dict().keys()
            ordinal_label = {k: i for i, k in enumerate(ordered_labels, 1)}
            self.ordinal_labels[variable] = ordinal_label

        return self

    def transform(self, X, y=None):

        for variable in self.variables:
            X[variable] = X[variable].map(self.ordinal_labels[variable])

        return X


class OrdinalEncoder:
    """Performs ordinal encoding on nonparametric features"""

    def __init__(self, variables, target):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.target_ = target
        self.labels_ = {}

    def fit(self, X, y):
        # persist encoding mapping to a dictionary

        tmp = pd.concat([X, y], axis=1)
        ranked = rankdata(tmp[self.target_])
        tmp["rank"] = ranked

        for feature in self.variables:
            ordered_labels = tmp.groupby([feature])["rank"].sum().sort_values().index
            ordinal_label = {k: i for i, k in enumerate(ordered_labels, 1)}
            self.labels_[feature] = ordinal_label

        return self

    def transform(self, X, y=None):

        for feature in self.variables:
            X[feature] = X[feature].map(self.labels_[feature])
        return X


class ContinuousScaler:
    """Scales and returns a chosen subset of continuous variables"""

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X, y=None):
        # learn and persist the mean and standard deviation
        # of the dataset

        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(X[self.variables])
        return self

    def transform(self, X, y=None):

        X[self.variables] = self.scaler_.transform(X[self.variables])
        return X
