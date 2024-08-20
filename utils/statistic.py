"""Class Statistic."""

import numpy as np
import math


def list_without_nan(func):
    """Decorator function wrapper to remove nan from list and return function.

    with the list without the nan
    """

    def wrapper(X):
        res = [x for x in X if not np.isnan(x)]
        return func(res)

    return wrapper


class statistic:
    """class which contains all the function of statistic."""

    @staticmethod
    @list_without_nan
    def count(X: list):
        """Return the number of value in the list."""
        return len(X)

    @staticmethod
    @list_without_nan
    def max(X: list):
        """Return max of the list."""
        if len(X) > 0:
            max_tmp = X[0]
            for x in X:
                if max_tmp < x:
                    max_tmp = x
            return max_tmp
        return 0

    @staticmethod
    @list_without_nan
    def min(X: list):
        """Return min of the list."""
        if len(X) > 0:
            min_tmp = X[0]
            for x in X:
                if min_tmp > x:
                    min_tmp = x
            return min_tmp
        return 0

    @staticmethod
    @list_without_nan
    def mean(X: list):
        """Return Mean of a list."""
        return sum(X) / len(X)

    @staticmethod
    def percentile(X: list, percent, key=lambda x: x):
        """Find the percentile of a list of values.

        take list  and percent between 0. and 1.
        """
        list_sorted = sort_without_nan(X)
        if not list_sorted:
            return None
        k = (len(list_sorted) - 1) * percent
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return key(list_sorted[int(k)])
        d0 = key(list_sorted[int(f)]) * (c - k)
        d1 = key(list_sorted[int(c)]) * (k - f)
        return d0 + d1

    @staticmethod
    @list_without_nan
    def f_quartile(X: list):
        """Return the first quartile (25) of a list."""
        X = sort_without_nan(X)
        len_x = len(X) + 1
        index = len_x * (1 / 4)
        return X[int(index)]

    @staticmethod
    @list_without_nan
    def s_quartile(X: list):
        """Return the second quartile (50) of a list."""
        X = sort_without_nan(X)
        len_x = len(X) + 1
        index = len_x * (2 / 4)
        return X[int(index)]

    @staticmethod
    @list_without_nan
    def t_quartile(X: list):
        """Return the third quartile (75) of a list."""
        X = sort_without_nan(X)
        len_x = len(X) + 1
        index = len_x * (3 / 4)
        return X[int(index)]

    @staticmethod
    @list_without_nan
    def std(X: list):
        """Return std of a list."""
        mean_lst = statistic.mean(X)
        variance = sum([((x - mean_lst) ** 2) for x in X]) / len(X)
        return variance**0.5

    @staticmethod
    @list_without_nan
    def sum(X: list):
        """Return Sum of a list."""
        sum_lst = 0
        for x in X:
            if not np.isnan(x):
                sum_lst += x
        return sum_lst


def sort_without_nan(X: list):
    """Return a sorted list without nan."""
    res = [x for x in X if not np.isnan(x)]
    return sorted(res)
