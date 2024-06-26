"""Percentile.py."""
import math
import numpy as np

def sort_without_nan(X: list):
   """Return a sorted list without nan."""
   res = [ x for x in X if not np.isnan(x)]
   return sorted(res)


def percentile(X:list, percent, key= lambda x:x ):
    """Find the percentile of a list of values.
    
    take list  and percent between 0. and 1.
    """
    list_sorted = sort_without_nan(X)
    if not list_sorted:
        return None
    k = (len(list_sorted)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(list_sorted[int(k)])
    d0 = key(list_sorted[int(f)]) * (c-k)
    d1 = key(list_sorted[int(c)]) * (k-f)
    return d0+d1