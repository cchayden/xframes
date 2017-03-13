class AggregatorPropertySet(object):
    """
    Store aggregator properties for one aggregator.

    XFrames comes with an assortment of aggregators, which are used
    in conjunction with xframes.groupby.  You can write your own aggregators
    if you need to.

    This class is used to define custom aggregators.  We will explain how by describing the
    builtin SUM function.

    You create a function that you use in the groupby call.  This function takes arguments, for
    instance the column name.  It returns an instance of AggregatorPropertySet and also a list
    of the arguments passed to it.  Let's examine SUM::

        def SUM(src_column):
            return AggregatorPropertySet(agg_sum, int, 'sum', 1), [src_column]

    In this example, SUM takes one argument, the name of the column to sum over.
    This argument is placed into an array and returned.

    The AggregatorPropertyset instance tells groupby how to do its work.  This includes
    the function to use to perform the aggregation (agg_sum), the type of the column to be created (int),
    the default name of the column if no name is given ('sum') and the number of
    arguments expected (1).

    The agg_sum function does the actual aggregation.  A simplified version of agg_sum
    is given below as an example.

    This function takes arguments:

        rows : A collection of rows.  Each row is a dictionary, in the form passed to xframes.apply.

        cols : A list of the arguments.  These are the values returned as the second member of SUM.

    Then the function (agg_sum) computes and returns the aggregated value.

    Here is the code for agg_sum::

        def agg_sum(rows, cols):
            src_col = cols[0]
            total = 0
            for row in rows:
                val = row[src_col]
                if not _is_missing(val):
                    total += val
            return total

    If you call SUM('some-col') in a groupby statement, then SUM returns the AggregatorPropertySet as shown above,
    and ['some-col'].

    Then the groupby command is executed.  For every distinct value of the grouped variable, it creates a row iterator
    and passes it to agg_sum, along with ['some-col'].  The number of rows may be very large, so they are not
    all computed and passed as a list, but are provided by an iterator.

    Then agg_sum extracts the desired value from each row (row[src_col]) and sums up
    the values.

    The function agg_sum is executed in a spark worker node, as with the function used in xframes.apply, and
    the same restrictions apply.
    """

    def __init__(self, agg_function, output_type, default_column_name, num_args):
        """
        Create a new instance.

        Parameters
        ----------
        agg_function: func(rows, cols)
            The agregator function.
            This is given a pyspark resultIterable produced by rdd.groupByKey
            and containing the rows matching a single group.
            It's responsibility is to compute and return the aggregate value for the group.

        output_type: type or int
            If a type is given, use that type as the output column type.
            If an integer is given, then the output type is the same as the
            input type of the column indexed by the integer.

        default_column_name: str
            The name of the aggregate column, if not supplied explicitly.

        num_args : int
            The number of arguments to the agg_function.

        """

        self.agg_function = agg_function
        self.default_column_name = default_column_name
        self.output_type = output_type
        self.num_args = num_args

    def get_output_type(self, input_type):
        candidate = self.output_type
        if isinstance(candidate, int):
            return input_type[candidate]
        return candidate


