
import traceback
import operator
import math
import datetime
import logging

from xframes.deps import HAS_MATPLOTLIB

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt

import xframes


class XPlot(object):
    """
    Plotting library for xFrames.

    Creates simple data plots.

    Parameters
    ----------
    axes : list, optional
        The size of the axes.  Should be a four-element list.
        [x_origin, y_origin, x_length, y_length]
        Defaults to [0.0, 0.0, 1.5, 1.0]

    alpha : float, optional
        The opacity of the plot.
    """
    def __init__(self, axes=None, alpha=None):
        """
        Create a plotting object.

        Parameters
        ----------
        axes : list, optional
            The size of the axes.  Should be a four-element list.
            [x_origin, y_origin, x_length, y_length]
            Defaults to [0.0, 0.0, 1.5, 1.0]

        alpha : float, optional
            The opacity of the plot.
        """
        self.axes = axes if axes else [0.0, 0.0, 1.5, 1.0]
        self.alpha = alpha or 0.5

    def make_barh(self, items, xlabel, ylabel, title=None):
        if not HAS_MATPLOTLIB:
            return
        if items is not None and len(items) > 0:
            try:
                y_pos = range(len(items))
                vals = [int(key[1]) for key in items]
                labels = [str(key[0])[:30] for key in items]

                def safe_decode(str):
                    try:
                        return str.decode('utf8')
                    except:
                        return 'string decde error'
                labels = [safe_decode(label) for label in labels]
                plt.barh(y_pos, vals, align='center', alpha=self.alpha)
                plt.yticks(y_pos, labels)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                if title:
                    plt.title(title)
                plt.show()
            except Exception as e:
                logging.warn("Make_barh: got an exception!")
                logging.warn(traceback.format_exc())
                logging.warn(e)

    # noinspection PyShadowingBuiltins
    def make_bar(self, items, xlabel, ylabel, title=None):
        if not HAS_MATPLOTLIB:
            return
        if items is not None:
            bins = len(items)
            try:
                counts = [col[1] for col in items]
                vals = [col[0] for col in items]
                x_pos = range(len(counts))
                plt.bar(x_pos, counts, align='center', alpha=self.alpha)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                delta = vals[1] - vals[0]
                min = vals[0]
                max = min + bins * delta
                if bins < 8:
                    n_ticks = bins
                else:
                    n_ticks = 8
                tick_delta = (max - min) / float(n_ticks)
                step = int(bins / float(n_ticks))
                if step <= 0: step = 1
                tick_pos = range(0, bins + 1, step)
                tick_labels = [min + i * tick_delta for i in range(n_ticks + 1)]
                tick_labels = [str(lab)[:5] for lab in tick_labels]
                plt.xticks(tick_pos, tick_labels)
                if title:
                    plt.title(title)
                plt.show()
            except Exception as e:
                logging.warn("Make_barh: got an exception!")
                logging.warn(traceback.format_exc())
                logging.warn(e)

    def top_values(self, xf, x_col, y_col, k=15, title=None, xlabel=None, ylabel=None):
        """
        Plot the top values of a column of data.

        Parameters
        ----------
        xf : XFrame
            An XFrame containing the columns to be plotted.

        x_col : str
            A column name: the top values in this column are plotted.  These values must be numerical.

        y_col : str
            A column name: the values in this colum will be used to label the corresponding values
            in the x column.

        k : int, optional
            The number of values to plot.  Defaults to 15.

        title : str, optional
            A plot title.

        xlabel : str, optional
            A label for the X axis.

        ylabel : str, optional
            A label for the Y axis.

        Examples
        --------
        (Come up with an example)
        """
        top_rows = xf.topk(x_col, k=k)
        items = [(row[y_col], row[x_col]) for row in top_rows]
        xlabel = xlabel or x_col
        ylabel = ylabel or y_col

        self.make_barh(items, xlabel, ylabel, title=title)

    def frequent_values(self, column, k=15, title=None, xlabel=None, ylabel=None,
                        epsilon=None, delta=None, num_items=None):
        """
        Plots the number of occurances of specific values in a column.  

        The most frequent values are plotted.

        Parameters
        ----------
        column : XArray
            The column to plot.  The number of distinct occurrances of each value is
            calculated and plotted.  

        k : int, optional
            The number of different values to plot.  Defaults to 15.

        title : str, optional
            A plot title.

        xlabel : str, optional
            A label for the X axis.

        ylabel : str, optional
            A label for the Y axis.

        epsilon : float, optional
            Governs accuracy of frequency counter.

        delta : float, optional
            Governs accuracy of frequency counter.

        num_items : float, optional
            Governs accuracy of frequency counter.

        Returns
        -------
        list of tuples
            List of (value, count) for the most frequent "k" values

        Examples
        --------
        (Need examples)

        """
        sk = column.sketch_summary()
        if epsilon:
            sk.set_frequency_sketch_parms(epsilon=epsilon)
        if delta:
            sk.set_frequency_sketch_parms(delta=delta)
        if num_items:
            sk.set_frequency_sketch_parms(num_items=num_items)

        fi = sk.frequent_items()
        if len(fi) > 0:
            sorted_fi = sorted(fi.iteritems(), key=operator.itemgetter(1), reverse=True)
        else:
            return []
        frequent = [x for x in sorted_fi[:k] if x[1] > 1]
        if len(frequent) > 0:
            xlabel = xlabel or 'Count'
            ylabel = ylabel or 'Value'
            title = title or "Frequent Values"
            self.make_barh(frequent, xlabel=xlabel, ylabel=ylabel, title=title)
        return frequent

    @staticmethod
    def create_histogram_buckets(vals, bins, min_val, max_val):
        if max_val == min_val:
            return None, None
        interval = max_val - min_val
        n_buckets = bins or 50
        bucket_vals = [0] * n_buckets
        usetd = isinstance(interval, datetime.timedelta)
        if usetd:
            delta = interval.total_seconds() / n_buckets
            for i in range(0, n_buckets):
                bucket_vals[i] = min_val + datetime.timedelta(seconds=(i * delta))
        else:
            delta = float(interval) / n_buckets
            for i in range(0, n_buckets):
                bucket_vals[i] = min_val + (i * delta)

        def iterate_values(value_iterator):
            bucket_counts = [0] * n_buckets
            for val in value_iterator:
                if val is None:
                    continue
                if isinstance(val, float ) and math.isnan(val):
                    continue
                if usetd:
                    b = int((val - min_val).total_seconds() / delta)
                else:
                    b = int((val - min_val) / delta)
                if b >= n_buckets:
                    b = n_buckets - 1
                elif b < 0:
                    b = 0
                bucket_counts[b] += 1
            yield bucket_counts

        def merge_accumulators(acc1, acc2):
            return [a1 + a2 for a1, a2 in zip(acc1, acc2)]

        accumulators = vals._impl._rdd.mapPartitions(iterate_values)
        bucket_counts = accumulators.reduce(merge_accumulators)
        return bucket_vals, bucket_counts

    def histogram(self,
                  column,
                  title=None,
                  bins=None,
                  sketch=None, 
                  xlabel=None, ylabel=None,
                  lower_cutoff=0.0, upper_cutoff=1.0,
                  lower_bound = None, upper_bound=None):
        """ 
        Plot a histogram.

        All values greater than the cutoff (given as a quantile) are set equal to the cutoff.

        Parameters
        ----------
        column : XArray
            A column to display.

        title : str, optional
            A plot title.

        bins : int, optional
            The number of bins to use.  Defaults to 50.

        sketch : Sketch, optional
            The column sketch.  If this is available, then it saves time not to recompute it.

        xlabel : str, optional
            A label for the X axis.

        ylabel : str, optional
            A label for the Y axis.

        lower_cutoff : float, optional
            This is a quantile value, between 0 and 1.  
            Values below this cutoff are placed in the first bin.
            Defaults to 0.

        upper_cutoff : float, optional
            This is a quantile value, between 0 and 1.  
            Values above this cutoff are placed in the last bin.
            Defaults to 1.0.

        lower_bound : float, optional
            Values below this bound are placed in the first bin.

        upper_bound : float, optional
            Values below this bound are placed in the last bin.

        bins : int, optional
            The number of bins to use.  Defaults to 50.

        Examples
        --------
        (Need examples)
        """
        if lower_cutoff < 0.0 or lower_cutoff > 1.0:
            raise ValueError('lower cutoff must be between 0.0 and 1.0')
        if upper_cutoff < 0.0 or upper_cutoff > 1.0:
            raise ValueError('upper cutoff must be between 0.0 and 1.0')
        if lower_cutoff >= upper_cutoff:
            raise ValueError('lower cutoff must be less than upper cutoff')

        bins = bins or 50
        sk = sketch or column.sketch_summary()
        q_epsilon = 0.01
        q_lower = None
        q_upper = None
        if lower_cutoff > 0.0:
            q_lower = float(sk.quantile(lower_cutoff)) - q_epsilon
        if upper_cutoff < 1.0:
            q_upper = float(sk.quantile(upper_cutoff)) + q_epsilon
        if lower_bound is not None:
            q_lower = lower_bound
        if upper_bound is not None:
            q_upper = upper_bound
        xlabel = xlabel or 'Value'
        ylabel = ylabel or 'Count'
        vals = column.dropna()

        def enforce_lower_cutoff(x):
            return max(x, q_lower)
        def enforce_upper_cutoff(x):
            return min(x, q_upper)
        if q_lower is not None:
            vals = vals.apply(enforce_lower_cutoff)
            hist_min = q_lower
        else:
            hist_min = sk.min()
        if q_upper is not None:
            vals = vals.apply(enforce_upper_cutoff)
            hist_max = q_upper
        else:
            hist_max = sk.max()
        bucket_counts, bucket_vals = self.create_histogram_buckets(vals, bins, hist_min, hist_max)
        column = [(x, y) for x, y in zip(bucket_counts, bucket_vals)]
        self.make_bar(column, xlabel=xlabel, ylabel=ylabel, title=title)

    def col_info(self, column, column_name=None, table_name=None, title=None, topk=None, bins=None, cutoff=False):
        """ 
        Print column summary information.

        The number of the most frequent values is shown.
        If the column to summarize is numerical or datetime, then a histogram is also shown.

        Parameters
        ----------
        column : XArray
            The column to summarize.

        column_name : str
            The column name.

        table_name : str, optional
            The table name; used to labeling only.  The table that us used for the data
            is given in the constructor.

        title : str, optional
            The plot title.

        topk: int, optional
            The number of frequent items to show.

        bins : int, optional
            The number of bins in a histogram.

        cutoff : float, optional
            The number to use as an upper cutoff, if the plot is a histogram.

        Examples
        --------
        (Need examples)
        """

        title = title or table_name
        column_name = column_name or ''
        table_name = table_name or ''
        print 'Table Name:  ', table_name
        print 'Column Name: ', column_name
        print 'Column Type: ', column.dtype().__name__
        sk = column.sketch_summary()
        print 'Rows:        ', sk.size()
        unique_items = sk.num_unique()
        print 'Unique Items:', unique_items
        print 'Approximate Frequent Items:'
        fi = sk.frequent_items()
        topk = topk or 15
        if len(fi) == 0:
            print '    None'
            top = None
        else:
            sorted_fi = sorted(fi.iteritems(), key=operator.itemgetter(1), reverse=True)
            top = [x for x in sorted_fi[:topk] if x[1] > 1]
            for key in top:
                print '   {:10}  {:10}'.format(key[1], key[0])
        col_type = column.dtype()
        if col_type is int or col_type is float:
            # number: show a histogram
            print 'Num Undefined:', sk.num_undefined()
            print 'Min:          ', sk.min()
            print 'Max:          ', sk.max()
            print 'Mean:         ', sk.mean()
            if unique_items > 1:
                print 'StDev:        ', sk.std()
                print 'Distribution Plot'
                upper_cutoff = cutoff or 1.0
                self.histogram(column, title=title, bins=bins, sketch=sk, upper_cutoff=upper_cutoff)

        if col_type is datetime.datetime:
            # datetime: show a histogram
            print 'Num Undefined:', sk.num_undefined()
            print 'Min:          ', sk.min()
            print 'Max:          ', sk.max()
            if unique_items > 1:
                print 'Distribution Plot'
                upper_cutoff = cutoff or 1.0
                self.histogram(column, title=title, bins=bins, sketch=sk, upper_cutoff=upper_cutoff)

        # ordinal: show a bar chart of frequent values
        # set x_col and y_col
        if top is not None:
            vals = xframes.XArray([key[0] for key in top], dtype=col_type)
            counts = xframes.XArray([key[1] for key in top], dtype=int)
            x_col = 'Count'
            y_col = column_name
            tmp = xframes.XFrame({x_col: counts, y_col: vals})
            tmp.show().top_values(x_col, y_col, title=title, k=topk)
