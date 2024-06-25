"""Percentile.py."""
import math

def percentile(X:list, percent, key= lambda x:x ):
    """Find the percentile of a list of values.
    
    take list  and percent between 0. and 1.
    """
    list_sorted = sorted(X)
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