"""
This module provides an implementation of XStream using pySpark RDDs.
"""
import json


from pyspark.streaming.kafka import KafkaUtils

from xframes.xobject_impl import XObjectImpl
from xframes.traced_object import TracedObject
from xframes.xframe import XFrame
from xframes.spark_context import CommonSparkContext
from xframes.lineage import Lineage
from xframes.util import safe_cast_val
import xframes
from xframes.xarray_impl import XArrayImpl


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


class XStreamImpl(XObjectImpl, TracedObject):
    """ Implementation for XStream. """

    def __init__(self, dstream=None, col_names=None, column_types=None, lineage=None):
        """
        Instantiate a XStream implementation.
        """
        self._entry()
        super(XStreamImpl, self).__init__(None)
        self._dstream = dstream
        col_names = col_names or []
        column_types = column_types or []
        self.col_names = list(col_names)
        self.column_types = list(column_types)
        self.lineage = lineage or Lineage.init_frame_lineage(Lineage.EMPTY, self.col_names)

    def _rv(self, dstream, col_names=None, column_types=None, lineage=None):
        """
        Return a new XFrameImpl containing the RDD, column names, column types, and lineage.

        Column names and types default to the existing ones.
        This is typically used when a function returns a new XFrame.
        """
        # only use defaults if values are None, not []
        col_names = self.col_names if col_names is None else col_names
        column_types = self.column_types if column_types is None else column_types
        lineage = lineage or self.lineage
        return XStreamImpl(dstream, col_names, column_types, lineage)

    @classmethod
    def create_from_text_files(cls, directory_path):
        cls._entry(directory_path=directory_path)
        ssc = CommonSparkContext.streaming_context()
        dstream = ssc.textFileStream(directory_path)
        return XStreamImpl(dstream=dstream, col_names=['line'], column_types=[str])

    @classmethod
    def create_from_socket_stream(cls, hostname, port):
        cls._entry(hostname=hostname, post=port)
        ssc = CommonSparkContext.streaming_context()
        dstream = ssc.socketTextStream(hostname, port)
        return XStreamImpl(dstream=dstream, col_names=['line'], column_types=[str])

    @classmethod
    def create_from_kafka_topics(cls, topics, kafka_servers, kafka_params):
        cls._entry(topics=topics, kafka_servers=kafka_servers, kafka_params=kafka_params)
        default_kafka_params = {'bootstrap.servers': kafka_servers}
        if kafka_params is not None:
            params = merge_dicts(default_kafka_params, kafka_params)
        else:
            params = default_kafka_params
        ssc = XObjectImpl.streaming_context()
        dstream = KafkaUtils.createDirectStream(ssc, topics, params)
        return XStreamImpl(dstream=dstream, col_names=['key', 'message'], column_types=[str, str])

    @classmethod
    def start(cls):
        cls._entry()
        ssc = CommonSparkContext.streaming_context()
        ssc.start()

    @classmethod
    def stop(cls, stop_spark_context, stop_gracefully):
        cls._entry(stop_spark_context=stop_spark_context, stop_gracefully=stop_gracefully)
        ssc = CommonSparkContext.streaming_context()
        ssc.stop(stop_spark_context, stop_gracefully)

    @classmethod
    def await_termination(cls, timeout):
        cls._entry(timeout=timeout)
        ssc = CommonSparkContext.streaming_context()
        if timeout is None:
            ssc.awaitTermination()
            return None
        return ssc.awaitTerminationOrTimeout(timeout)

    def column_names(self):
        self._entry()
        return self.col_names

    def dtype(self):
        self._entry()
        return self.column_types

    def lineage_as_dict(self):
        self._entry()
        return {'table': self.lineage.table_lineage,
                'column': self.lineage.column_lineage}

    def transform_row(self, row_fn, column_names, column_types):
        self._entry(column_names=column_names)
        col_names = self.col_names    # dereference col_names outside lambda
        column_names = column_names or col_names
        column_types = column_types or self.column_types

        # fn needs the row as a dict
        def build_row(names, row):
            return dict(zip(names, row))

        def transformer(row):
            return row_fn(build_row(col_names, row))

        res = self._dstream.map(transformer)
        return self._rv(res, column_names, column_types)

    def count(self):
        self._entry()
        res = self._dstream.count().map(lambda c: (c,))
        return self._rv(res, ['count'], [int])

    def count_distinct(self, col):
        self._entry(col=col)
        if col not in self.col_names:
            raise ValueError("Column name does not exist: '{}'.".format(col))
        index = self.column_names().index(col)
        dstream = self._dstream.map(lambda row: row[index])
        res = dstream.countByValue().map(lambda c: (c,))
        return self._rv(res, ['count'], [int])

    def flat_map(self, fn, column_names, column_types):
        self._entry(column_names=column_names, column_types=column_types)
        names = self.col_names

        # fn needs the row as a dict
        def build_row(names, row):
            return dict(zip(names, row))
        res = self._dstream.flatMap(lambda row: fn(build_row(names, row)))
        res = res.map(tuple)
        lineage = self.lineage.flat_map(column_names, names)
        return self._rv(res, column_names, column_types, lineage)

    def apply(self, fn, dtype):
        """
        Transform each XFrame in the XStream to an XArray according to a
        specified function. Returns a array RDD of ``dtype`` where each element
        in this array RDD is transformed by `fn(x)` where `x` is a single row in
        the xframe represented as a dictionary.  The ``fn`` should return
        exactly one value which is or can be cast into type ``dtype``.
        """
        self._entry(dtype=dtype)
        names = self.col_names

        # fn needs the row as a dict
        def build_row(names, row):
            return dict(zip(names, row))

        def transformer(row):
            result = fn(build_row(names, row))
            if not isinstance(result, dtype):
                return safe_cast_val(result, dtype)
            return result
        res = self._dstream.map(transformer)
        lineage = self.lineage.apply(names)
        return xframes.xarray_impl.XArrayImpl(res, dtype, lineage)

    def transform_col(self, col, fn, dtype):
        """
        Transform a single column according to a specified function.
        The remaining columns are not modified.
        The type of the transformed column types becomes `dtype`, with
        the new value being the result of `fn(x)`, where `x` is a single row in
        the XFrame represented as a dictionary.  The `fn` should return
        exactly one value which can be cast into type `dtype`.

        Parameters
        ----------
        col : string
            The name of the column to transform.

        fn : function, optional
            The function to transform each row of the XFrame. The return
            type should be convertible to `dtype`
            If the function is not given, an identity function is used.

        dtype : dtype, optional
            The column data type of the new XArray. If None, the first 100
            elements of the array are used to guess the target
            data type.

        Returns
        -------
        out : XStream
            An XStream with the given column transformed by the function and cast to the given type.
        """

        self._entry(col=col)
        if col not in self.col_names:
            raise ValueError("Column name does not exist: '{}'.".format(col))
        col_index = self.col_names.index(col)
        col_names = self.col_names     # dereference outside lambda

        # fn needs the row as a dict
        def build_row(names, row):
            return dict(zip(names, row))

        def transformer(row):
            result = fn(build_row(col_names, row))
            if not isinstance(result, dtype):
                result = safe_cast_val(result, dtype)
            lst = list(row)
            lst[col_index] = result
            return tuple(lst)

        new_col_types = list(self.column_types)
        new_col_types[col_index] = dtype

        res = self._rdd.map(transformer)
        return self._rv(res, col_names, new_col_types)

    def filter_by_function(self, fn, column_name, exclude):
        """
        Perform filtering on a single column by a function
        """
        col_index = self.col_names.index(column_name)

        def filter_fun(row):
            res = fn(row[col_index])
            return not res if exclude else res

        res = self._dstream.filter(filter_fun)
        return self._rv(res)

    def filter_by_function_row(self, fn, exclude):
        """
        Perform filtering on all columns by a function
        """
        # fn needs the row as a dict
        col_names = self.col_names

        def filter_fun(row):
            res = fn(dict(zip(col_names, row)))
            return not res if exclude else res

        res = self._dstream.filter(filter_fun)
        return self._rv(res)

    #####################
    #  Output Operations
    #####################

    def process_rows(self, row_fn, init_fn, final_fn):
        self._entry()

        def process_rdd_rows(rdd):
            xf = XFrame.from_rdd(rdd, column_names=self.col_names)
            xf.foreach(row_fn, init_fn, final_fn)

        self._dstream.foreachRDD(process_rdd_rows)

    def save(self, prefix, suffix):
        self._entry(prefix=prefix, suffix=suffix)
        self._dstream.saveAsTextFiles(prefix, suffix)

    def print_frames(self, num_rows, num_columns,
                   max_column_width, max_row_width,
                   wrap_text, max_wrap_rows, footer):
        def print_rdd_rows(rdd):
            xf = XFrame.from_rdd(rdd, column_names=self.col_names)
            xf.print_rows(num_rows, num_columns, max_column_width, max_row_width,
                         wrap_text, max_wrap_rows, footer)

        self._dstream.foreachRDD(print_rdd_rows)
