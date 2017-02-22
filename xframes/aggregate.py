
from aggregator_property_set import AggregatorPropertySet

# Builtin aggregators for groupby
from aggregator_impl import agg_sum, agg_argmax, agg_argmin, agg_max, agg_min, \
    agg_count, agg_mean, agg_variance, agg_stdv, agg_select_one, \
    agg_concat_list, agg_concat_dict, agg_values, agg_values_count, agg_quantile


# noinspection PyPep8Naming
def SUM(src_column):
    """
    Builtin sum aggregator for groupby

    Examples
    --------

    Get the sum of the rating column for each user.
    >>> xf.groupby("user", {'rating_sum':aggregate.SUM('rating')})

    """
    return AggregatorPropertySet(agg_sum, int, 'sum', 1), [src_column]


# noinspection PyPep8Naming
def ARGMAX(agg_column, out_column):
    """
    Builtin arg maximum aggregator for groupby.

    Examples
    --------

    Get the movie with maximum rating per user.

    >>> xf.groupby("user",
                    {'best_movie':aggregate.ARGMAX('rating','movie')})
    """
    return AggregatorPropertySet(agg_argmax, 1, 'argmax', 2), [agg_column, out_column]


# noinspection PyPep8Naming
def ARGMIN(agg_column, out_column):
    """
    Builtin arg minimum aggregator for groupby.

    Examples
    --------

    Get the movie with minimum rating per user.

    >>> xf.groupby("user",
                    {'best_movie':aggregate.ARGMIN('rating','movie')})

    """
    return AggregatorPropertySet(agg_argmin, 1, 'argmin', 2), [agg_column, out_column]


# noinspection PyPep8Naming
def MAX(src_column):
    """
    Builtin maximum aggregator for groupby

    Examples
    --------

    Get the maximum rating of each user.

    >>> xf.groupby("user",
                    {'rating_max':aggregate.MAX('rating')})

    """
    return AggregatorPropertySet(agg_max, 0, 'max', 1), [src_column]


# noinspection PyPep8Naming
def MIN(src_column):
    """
    Builtin minimum aggregator for groupby

    Examples
    --------

    Get the minimum rating of each user.

    >>> xf.groupby("user",
                    {'rating_min':aggregate.MIN('rating')})

    """
    return AggregatorPropertySet(agg_min, 0, 'min', 1), [src_column]


# noinspection PyPep8Naming
def COUNT():
    """
    Builtin count aggregator for groupby

    Examples
    --------

    Get the number of occurrences of each user.

    >>> xf.groupby("user",
                    {'count':aggregate.COUNT()})

    """
    return AggregatorPropertySet(agg_count, int, 'count', 0), ['']


# noinspection PyPep8Naming
def MEAN(src_column):
    """
    Builtin average aggregator for groupby.

    Examples
    --------

    Get the average rating of each user.

    >>> xf.groupby("user",
                    {'rating_mean':aggregate.MEAN('rating')})
    """
    return AggregatorPropertySet(agg_mean, float, 'mean', 1), [src_column]


# noinspection PyPep8Naming
def VARIANCE(src_column):
    """
    Builtin variance aggregator for groupby.

    Examples
    --------

    Get the rating variance of each user.

    >>> xf.groupby("user",
                 {'rating_var':aggregate.VARIANCE('rating')})

    """
    return AggregatorPropertySet(agg_variance, float, 'variance', 1), [src_column]


# noinspection PyPep8Naming
def STDV(src_column):
    """
    Builtin standard deviation aggregator for groupby.

    Examples
    --------

    Get the rating standard deviation of each user.

    >>> xf.groupby("user",
                    {'rating_stdv':aggregate.STDV('rating')})

    """
    return AggregatorPropertySet(agg_stdv, float, 'stdv', 1), [src_column]


# noinspection PyPep8Naming
def SELECT_ONE(src_column):
    """
    Builtin aggregator for groupby which selects one row in the group.

    Examples
    --------

    Get one rating row from a user.

    >>> xf.groupby("user", {'rating':aggregate.SELECT_ONE('rating')})

    If multiple columns are selected, they are guaranteed to come from the
    same row. For instance:
    >>> xf.groupby("user", {'rating':aggregate.SELECT_ONE('rating'), 'item':aggregate.SELECT_ONE('item')})

    The selected 'rating' and 'item' value for each user will come from the
    same row in the XFrame.
    """

    # use seed to make selection repeatable
    # it would be more random to use the column name
    seed = src_column
    return AggregatorPropertySet(agg_select_one, 0, 'select-one', 1), [src_column, seed]


# noinspection PyPep8Naming
def CONCAT(src_column, dict_value_column=None):
    """
    Builtin aggregator that combines values from one or two columns in one group
    into either a dictionary value, list value or array value.

    Examples
    --------

    To combine values from two columns that belong to one group into
    one dictionary value:

    >>> xf.groupby(["document"],
                   {"word_count": aggregate.CONCAT("word", "count")})

    To combine values from one column that belong to one group into a list value:

    >>> xf.groupby(["user"],
                   {"friends": aggregate.CONCAT("friend")})

    """
    if dict_value_column is None:
        return AggregatorPropertySet(agg_concat_list, list, 'concat', 1), [src_column]
    else:
        return AggregatorPropertySet(agg_concat_dict, dict, 'concat', 1), [src_column, dict_value_column]


# noinspection PyPep8Naming
def VALUES(src_column):
    """
    Builtin aggregator that combines distinct values from one  column in one group
    into a list value.

    Examples
    --------

    To combine values from one column that belong to one group into a list value:

    >>> xf.groupby(["user"],
                     {"friends": aggregate.VALUES("friend")})

    """
    return AggregatorPropertySet(agg_values, list, 'values', 1), [src_column]


# noinspection PyPep8Naming
def VALUES_COUNT(src_column):
    """
    Builtin aggregator that combines distinct values from one  column in one group
    into a dictionary value of unique values and their counts.

    Examples
    --------

    To combine values from one column that belong to one group into a dictionary of friend: count values:

    >>> xf.groupby(["user"],
       {"friends": aggregate.VALUES_COUNT("friend")})

    """
    return AggregatorPropertySet(agg_values_count, dict, 'values-count', 1), [src_column]


# noinspection PyPep8Naming
def QUANTILE(src_column, *args):
    """
    Builtin approximate quantile aggregator for groupby.
    Accepts as an argument, one or more of a list of quantiles to query.

    Examples
    --------

    To extract the median
        >>> xf.groupby("user",
                        {'rating_quantiles': aggregate.QUANTILE('rating', 0.5)})

    To extract a few quantiles
        >>> xf.groupby("user",
                        {'rating_quantiles': aggregate.QUANTILE('rating', [0.25,0.5,0.75])})

    Or equivalently
        >>> xf.groupby("user",
                        {'rating_quantiles': aggregate.QUANTILE('rating', 0.25,0.5,0.75)})

    The returned quantiles are guaranteed to have 0.5% accuracy. That is to say,
    if the requested quantile is 0.50, the resultant quantile value may be
    between 0.495 and 0.505 of the true quantile.
    """
    if len(args) == 1:
        quantiles = args[0]
    else:
        quantiles = list(args)

    if not hasattr(quantiles, '__iter__'):
        quantiles = [quantiles]
    query = ",".join([str(i) for i in quantiles])
    return AggregatorPropertySet(agg_quantile, float, 'quantile', 1), [src_column], '[' + query + ']'
