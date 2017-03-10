"""
This module provides an implementation of XStream using pySpark RDDs.
"""

from pyspark.streaming.kafka import KafkaUtils

from xframes.xobject_impl import XObjectImpl
from xframes.traced_object import TracedObject
from xframes.xframe import XFrame
from xframes.spark_context import CommonSparkContext
from xframes.lineage import Lineage
from xframes.type_utils import safe_cast_val
from xframes.utils import merge_dicts
from xframes.xobject_impl import UnimplementedException


# TODO: move back into xframes.utils
def build_row(names, row, use_columns=None, use_columns_index=None):
    if use_columns:
        names = [name for name in names if name in use_columns]
        row = [row[i] for i in use_columns_index]
    return dict(zip(names, row))


class XStreamImpl(XObjectImpl, TracedObject):
    """ Implementation for XStream. """

    def __init__(self, dstream=None, col_names=None, column_types=None, lineage=None):
        """
        Instantiate a XStream implementation.
        """
        self._entry()
        super(XStreamImpl, self).__init__()
        self._dstream = dstream
        col_names = col_names or []
        column_types = column_types or []
        self.col_names = list(col_names)
        self.column_types = list(column_types)
        self.lineage = lineage or Lineage.init_frame_lineage(Lineage.EMPTY, self.col_names)

    def _replace_dstream(self, dstream):
        self._dstream = self._wrap_rdd(dstream)

    def dump_debug_info(self):
        return self._dstream.toDebugString()

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
    def set_checkpoint(cls, checkpoint_dir):
        cls._entry(clscheckpoint_dir=checkpoint_dir)
        ssc = CommonSparkContext.streaming_context()
        ssc.checkpoint(checkpoint_dir)

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

    def to_dstream(self, number_of_partitions=None):
        """
        Returns the underlying DStream.

        Discards the column name and type information.
        """
        self._entry(number_of_partitions=number_of_partitions)
        return self._dstream.repartition(number_of_partitions) if number_of_partitions is not None else self._dstream

    def transform_row(self, row_fn, column_names, column_types):
        self._entry(column_names=column_names)
        col_names = self.col_names    # dereference col_names outside lambda
        column_names = column_names or col_names
        column_types = column_types or self.column_types

        def transformer(row):
            return row_fn(build_row(col_names, row))

        res = self._dstream.map(transformer)
        return self._rv(res, column_names, column_types)

    def num_rows(self):
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

        res = self._dstream.flatMap(lambda row: fn(build_row(names, row)))
        res = res.map(tuple)
        lineage = self.lineage.flat_map(column_names, names)
        return self._rv(res, column_names, column_types, lineage)

    def apply(self, fn, dtype):
        self._entry(dtype=dtype)
        names = self.col_names

        def transformer(row):
            result = fn(build_row(names, row))
            if not isinstance(result, dtype):
                return safe_cast_val(result, dtype)
            return result
        res = self._dstream.map(transformer)
        lineage = self.lineage.apply(names)
        # TODO: this is not right -- we need to distinguish between tuples and simple values
        return self._rv(res, ['value'], [dtype], lineage)

    def transform_col(self, col, fn, dtype):
        self._entry(col=col)
        if col not in self.col_names:
            raise ValueError("Column name does not exist: '{}'.".format(col))
        col_index = self.col_names.index(col)
        col_names = self.col_names     # dereference outside lambda

        def transformer(row):
            result = fn(build_row(col_names, row))
            if not isinstance(result, dtype):
                result = safe_cast_val(result, dtype)
            lst = list(row)
            lst[col_index] = result
            return tuple(lst)

        new_col_types = list(self.column_types)
        new_col_types[col_index] = dtype

        res = self._dstream.map(transformer)
        return self._rv(res, col_names, new_col_types)

    def filter(self, values, column_name, exclude):
        col_index = self.col_names.index(column_name)

        def filter_fun(row):
            val = row[col_index]
            return val not in values if exclude else val in values

        res = self._dstream.filter(filter_fun)
        return self._rv(res)

    def filter_by_function(self, fn, column_name, exclude):
        col_index = self.col_names.index(column_name)

        def filter_fun(row):
            filtered = fn(row[col_index])
            return not filtered if exclude else filtered

        res = self._dstream.filter(filter_fun)
        return self._rv(res)

    def filter_by_function_row(self, fn, exclude):

        col_names = self.col_names

        def filter_fun(row):
            filtered = fn(dict(zip(col_names, row)))
            return not filtered if exclude else filtered

        res = self._dstream.filter(filter_fun)
        return self._rv(res)

    def update_state(self, fn, col_name, initial_state):

        def generator(initial_state):
            elems_at_a_time = 200000
            initial_state._impl.begin_iterator()
            ret = initial_state._impl.iterator_get_next(elems_at_a_time)
            while True:
                for j in ret:
                    # Iterator returns tuples
                    yield j

                if len(ret) == elems_at_a_time:
                    ret = initial_state._impl.iterator_get_next(elems_at_a_time)
                else:
                    break

        state_column_names = initial_state.column_names()
        state_column_types = initial_state.column_types()

        initial_state_dict = {}
        index = self.col_names.index(col_name)
        for row in generator(initial_state):
            initial_state_dict[row[index]] = row

        names = self.column_names()

        def update_fn(events, state):
            if len(events) == 0:
                return state
            key = events[0][col_name]
            return fn(events, state, initial_state_dict.get(key, None))

        keyed_dstream = self._dstream.map(lambda row: (row[index], build_row(names, row)))
        res = keyed_dstream.updateStateByKey(update_fn)
        #res = res.flatMap(lambda kv: kv[1])
        res = res.map(lambda kv: kv[1])
        return self._rv(res, state_column_names, state_column_types)

    # noinspection PyMethodMayBeStatic
    def select_column(self, column_name):
        """
        Get the array RDD that corresponds with
        the given column_name as an XArray.
        """
        self._entry(column_name=column_name)
        if column_name not in self.col_names:
            raise ValueError("Column name does not exist: '{} in {}'.".format(column_name, self.col_names))

        col = self.col_names.index(column_name)
        res = self._dstream.map(lambda row: (row[col], ))
        col_type = self.column_types[col]
        lineage = self.lineage              # <=== just a guess
        return self._rv(res, [column_name], [col_type])

    # noinspection PyMethodMayBeStatic
    def select_columns(self, keylist):
        """
        Creates RDD composed only of the columns referred to in the given list of
        keys, as an XFrame.
        """
        self._entry(keylist=keylist)

        def get_columns(row, cols):
            return tuple([row[col] for col in cols])
        cols = [self.col_names.index(key) for key in keylist]
        names = [self.col_names[col] for col in cols]
        types = [self.column_types[col] for col in cols]
        res = self._dstream.map(lambda row: get_columns(row, cols))
        lineage = self.lineage.select_columns(names)
        return self._rv(res, names, types, lineage)

    # noinspection PyMethodMayBeStatic
    def add_column(self, values, col_name):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def add_columns_frame(self, xf_impl):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def add_columns_array(self, cols_impl, namelist):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def replace_selected_column(self, name, cols_impl):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def remove_column(self, name):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def remove_columns(self, column_names):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def swap_columns(self, column_1, column_2):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def reorder_columns(self, column_names):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def replace_column_names(self, new_names):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def groupby_aggregate(self, key_columns_array, group_columns, group_output_columns, group_properties):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def copy_range(self, start, step, stop):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def add_column_const_in_place(self, name, value):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def add_column_in_place(self, col, name):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def replace_single_column_in_place(self, column_name, col):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def replace_selected_column_in_place(self, column_name, col):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def remove_column_in_place(self, name):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def replace_column_const_in_place(self, name, value):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def add_columns_array_in_place(self, cols, namelist):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def add_columns_frame_in_place(self, other):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def join(self, right, how, join_keys):
        raise UnimplementedException()

    #####################
    #  Output Operations
    #####################

    def process_rows(self, row_fn, init_fn, final_fn):
        self._entry()
        column_names = self.col_names

        def process_rdd_rows(rdd):
            xf = XFrame.from_rdd(rdd, column_names=column_names)
            xf.foreach(row_fn, init_fn, final_fn)

        self._dstream.foreachRDD(process_rdd_rows)

    def process_frames(self, row_fn, init_fn, final_fn):
        self._entry()
        column_names = self.col_names
        init_val = init_fn() if init_fn is not None else None

        def process_rdd_frame(rdd):
            xf = XFrame.from_rdd(rdd, column_names=column_names)
            row_fn(xf, init_val)

        self._dstream.foreachRDD(process_rdd_frame)
        if final_fn is not None:
            final_fn()

    def save(self, prefix, suffix):
        self._entry(prefix=prefix, suffix=suffix)
        self._dstream.saveAsTextFiles(prefix, suffix)

    def print_frames(self, num_rows, num_columns,
                     max_column_width, max_row_width,
                     wrap_text, max_wrap_rows, footer):
        column_names = self.column_names()      # copy reference outside function

        def print_rdd_rows(rdd):
            xf = XFrame.from_rdd(rdd, column_names=column_names)
            xf.print_rows(num_rows, num_columns, max_column_width, max_row_width,
                          wrap_text, max_wrap_rows, footer)

        self._dstream.foreachRDD(print_rdd_rows)
