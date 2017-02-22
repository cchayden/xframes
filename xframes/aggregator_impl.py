"""
This module provides aggregator properties, used to define aggregators for groupby.
"""

import random
import math
from collections import Counter


def _is_missing(x):
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False


# Each of these functions operates on a pyspark resultIterable
#  produced by groupByKey and directly produces the aggregated result.

# All of them skip overrr missing values


def _collect_non_missing(rows, src_col, out_col=None):
    out_col = out_col or src_col
    return [row[out_col] for row in rows if not _is_missing(row[src_col])]


def agg_sum(rows, cols):
    src_col = cols[0]
    vals = _collect_non_missing(rows, src_col)
    if len(vals) == 0:
        return 0
    return sum(vals)


def agg_argmax(rows, cols): 
    agg_col = cols[0]
    out_col = cols[1]
    vals = _collect_non_missing(rows, agg_col)
    if len(vals) == 0:
        return None
    row_index = vals.index(max(vals))
    vals = _collect_non_missing(rows, agg_col, out_col)
    if len(vals) == 0:
        return None
    return vals[row_index]


def agg_argmin(rows, cols): 
    agg_col = cols[0]
    out_col = cols[1]
    vals = _collect_non_missing(rows, agg_col)
    if len(vals) == 0:
        return None
    row_index = vals.index(min(vals))
    vals = _collect_non_missing(rows, agg_col, out_col)
    if len(vals) == 0:
        return None
    return vals[row_index]


def agg_max(rows, cols): 
    # cols: [src_col]
    vals = _collect_non_missing(rows, cols[0])
    if len(vals) == 0:
        return None
    return max(vals)


def agg_min(rows, cols): 
    src_col = cols[0]
    vals = _collect_non_missing(rows, src_col)
    if len(vals) == 0:
        return None
    return min(vals)


# noinspection PyUnusedLocal
def agg_count(rows, cols):
    # Missing values do not matter here.
    return len(rows)


def agg_mean(rows, cols):
    src_col = cols[0]
    vals = _collect_non_missing(rows, src_col)
    if len(vals) == 0:
        return None
    return sum(vals) / float(len(vals))


def agg_variance(rows, cols):
    src_col = cols[0]
    vals = _collect_non_missing(rows, src_col)
    if len(vals) == 0:
        return None
    avg = sum(vals) / float(len(vals))
    return sum([(avg - val) ** 2 for val in vals]) / float(len(vals))


def agg_stdv(rows, cols):
    variance = agg_variance(rows, cols)
    if variance is None:
        return None
    return math.sqrt(variance)


def agg_select_one(rows, cols):
    src_col = cols[0]
    seed = cols[1]
    vals = _collect_non_missing(rows, src_col)
    num_vals = len(vals)
    if num_vals == 0:
        return None
    random.seed(seed)
    row_index = random.randint(0, num_vals - 1)
    val = vals[row_index]
    return val


def agg_concat_list(rows, cols): 
    src_col = cols[0]
    return _collect_non_missing(rows, src_col)


def agg_concat_dict(rows, cols): 
    src_col = cols[0]
    dict_value_col = cols[1]
    def collect_non_missing_kv(key_col, val_col):
        non_missing_rows = [row for row in rows if not _is_missing(row[key_col])]
        if len(non_missing_rows) == 0:
            return {}
        return {row[key_col]: row[val_col] for row in non_missing_rows}

    return collect_non_missing_kv(src_col, dict_value_col)


def agg_values(rows, cols):
    src_col = cols[0]
    def collect_values_non_missing(src_col, out_col=None):
        out_col = out_col or src_col
        return list({row[out_col] for row in rows if not _is_missing(row[src_col])})

    return collect_values_non_missing(src_col)


def agg_values_count(rows, cols):
    src_col = cols[0]
    def collect_values_count_non_missing(src_col, out_col=None):
        out_col = out_col or src_col
        return dict(Counter([row[out_col] for row in rows if not _is_missing(row[src_col])]))

    return collect_values_count_non_missing(src_col)


# noinspection PyUnusedLocal
def agg_quantile(rows, cols):
    # cols: [src_col, quantile]
    # cols: [src_col, [quantile ...]]
    # not imlemented
    return None
