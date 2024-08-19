"""Separate data per feature per house."""

from pandas.api.types import is_numeric_dtype


def separate_data_per_feature_per_house(data_csv):
    """Separate data csv in dict \
 
    for key the name of feature and value a dict  with key house name and value the result of the student.
    """
    dict_per_features = {}
    houses = data_csv["Hogwarts House"].unique()
    list_features = list(
        [x for x in data_csv.columns[1:] if is_numeric_dtype(data_csv[x])]
    )
    for features in list_features:
        dict_per_features[features] = {}
        for house in houses:
            dict_per_features[features][house] = list(
                data_csv[data_csv["Hogwarts House"] == house][features]
            )
    return dict_per_features
