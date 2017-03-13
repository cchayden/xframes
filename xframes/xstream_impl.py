"""
This module provides an implementation of XStream using pySpark RDDs.
"""
import logging
import copy

from xframes.traced_object import TracedObject
from xframes.xframe import XFrame
from xframes.spark_context import CommonSparkContext
from xframes.lineage import Lineage
from xframes.type_utils import safe_cast_val
from xframes.utils import merge_dicts
from xframes.object_utils import wrap_rdd
from xframes.object_utils import UnimplementedException


# TODO: move back into xframes.utils
def build_row(names, row, use_columns=None, use_columns_index=None):
    if use_columns:
        names = [name for name in names if name in use_columns]
        row = [row[i] for i in use_columns_index]
    return dict(zip(names, row))


class XStreamImpl(TracedObject):
    """ Implementation for XStream. """

    def __init__(self, dstream=None, column_names=None, column_types=None, lineage=None):
        """
        Instantiate a XStream implementation.
        """
        self._entry()
        super(XStreamImpl, self).__init__()
        self._dstream = dstream
        column_names = column_names or []
        column_types = column_types or []
        self.col_names = list(column_names)
        self.column_types = list(column_types)
        self.lineage = lineage or Lineage.init_frame_lineage(Lineage.EMPTY, self.col_names)

    def _replace_dstream(self, dstream):
        self._dstream = wrap_rdd(dstream)

    def dump_debug_info(self):
        return self._dstream.toDebugString()

    def _rv(self, dstream, column_names=None, column_types=None, lineage=None):
        """
        Return a new XFrameImpl containing the RDD, column names, column types, and lineage.

        Column names and types default to the existing ones.
        This is typically used when a function returns a new XFrame.
        """
        # only use defaults if values are None, not []
        column_names = self.col_names if column_names is None else column_names
        column_types = self.column_types if column_types is None else column_types
        lineage = lineage or self.lineage
        return XStreamImpl(dstream, column_names, column_types, lineage)

    def _replace(self, dstream, column_names=None, column_types=None, lineage=None):
        """
        Replaces the existing DStream, column names, column types, and lineage with new values.

        Column names, types, and lineage default to the existing ones.
        This is typically used when a function modifies the current XFrame.
        """
        self._dstream = dstream
        if column_names is not None:
            self.col_names = column_names
        if column_types is not None:
            self.column_types = column_types
        if lineage is not None:
            self.lineage = lineage

        self._num_rows = None
        self.materialized = False
        return self

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
        return XStreamImpl(dstream=dstream, column_names=['line'], column_types=[str])

    @classmethod
    def create_from_socket_stream(cls, hostname, port):
        cls._entry(hostname=hostname, post=port)
        ssc = CommonSparkContext.streaming_context()
        dstream = ssc.socketTextStream(hostname, port)
        return XStreamImpl(dstream=dstream, column_names=['line'], column_types=[str])

    @classmethod
    def create_from_kafka_topic(cls, topics, kafka_servers, kafka_params):
        cls._entry(topics=topics, kafka_servers=kafka_servers, kafka_params=kafka_params)
        from pyspark.streaming.kafka import KafkaUtils
        default_kafka_params = {'bootstrap.servers': kafka_servers}
        if kafka_params is not None:
            params = merge_dicts(default_kafka_params, kafka_params)
        else:
            params = default_kafka_params
        ssc = CommonSparkContext.streaming_context()
        dstream = KafkaUtils.createDirectStream(ssc, topics, params)
        return XStreamImpl(dstream=dstream, column_names=['key', 'message'], column_types=[str, str])

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

    def num_columns(self):
        self._entry()
        return len(self.col_names)

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

    def set_checkpoint_interval(self, interval):
        self._dstream.checkpoint(interval)

    def transform_row(self, row_fn, column_names, column_types):
        self._entry(column_names=column_names, column_types=column_types)
        input_column_names = self.col_names

        def transformer(row):
            return row_fn(build_row(input_column_names, row))

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
        names = self.col_names

        def transformer(row):
            result = fn(build_row(names, row))
            if not isinstance(result, dtype):
                result = safe_cast_val(result, dtype)
            lst = list(row)
            lst[col_index] = result
            return tuple(lst)

        new_col_types = list(self.column_types)
        new_col_types[col_index] = dtype

        res = self._dstream.map(transformer)
        return self._rv(res, names, new_col_types)

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

        names = self.col_names

        def filter_fun(row):
            filtered = fn(dict(zip(names, row)))
            return not filtered if exclude else filtered

        res = self._dstream.filter(filter_fun)
        return self._rv(res)

    def update_state(self, fn, column_name, initial_state):
        names = self.column_names()
        index = self.col_names.index(column_name)

        def generator(init_state):
            elems_at_a_time = 200000
            init_state.impl().begin_iterator()
            ret = init_state.impl().iterator_get_next(elems_at_a_time)
            while True:
                for j in ret:
                    # Iterator returns tuples
                    yield j

                if len(ret) == elems_at_a_time:
                    ret = init_state.impl().iterator_get_next(elems_at_a_time)
                else:
                    break

        def make_initial_state_dict(init_state):
            init_dict = {}
            for row in generator(init_state):
                init_dict[row[index]] = row
            return init_dict

        state_column_names = initial_state.column_names()
        state_column_types = initial_state.column_types()
        initial_state_dict = make_initial_state_dict(initial_state)

        def update_fn(events, state):
            if len(events) == 0:
                return state
            key = events[0][column_name]
            return fn(events, state, initial_state_dict.get(key, None))

        keyed_dstream = self._dstream.map(lambda row: (row[index], build_row(names, row)))
        res = keyed_dstream.updateStateByKey(update_fn)
        res = res.map(lambda kv: kv[1])
        return self._rv(res, state_column_names, state_column_types)

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
        column_type = self.column_types[col]
        return self._rv(res, [column_name], [column_type])

    def select_columns(self, keylist):
        """
        Creates RDD composed only of the columns referred to in the given list of
        keys, as an XFrame.
        """
        self._entry(keylist=keylist)

        cols = [self.col_names.index(key) for key in keylist]
        names = [self.col_names[col] for col in cols]

        def get_columns(row):
            return tuple([row[col] for col in cols])
        types = [self.column_types[col] for col in cols]
        res = self._dstream.map(get_columns)
        lineage = self.lineage.select_columns(names)
        return self._rv(res, names, types, lineage)

    def copy(self):
        """
        Creates a copy of the XStreamImpl.

        The underlying DStream is immutale, so we just need to copy the metadata.
        """
        self._entry()
        return self._rv(self._dstream)

    # noinspection PyMethodMayBeStatic
    def add_column(self, col, name):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def add_column_in_place(self, col, name):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def add_columns_array(self, cols, namelist):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def add_columns_array_in_place(self, cols, namelist):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def add_columns_frame(self, other):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def add_columns_frame_in_place(self, other):
        raise UnimplementedException()

    def remove_column(self, name):
        """
        Remove a column from the XStream.

        This operation creates a new XStreamImpl and returns it.
        """
        self._entry(name=name)
        col = self.col_names.index(name)
        names = copy.copy(self.col_names)
        column_types = copy.copy(self.column_types)
        names.pop(col)
        column_types.pop(col)

        def pop_col(row):
            lst = list(row)
            lst.pop(col)
            return tuple(lst)
        res = self._dstream.map(pop_col)
        lineage = self.lineage.remove_column(name)
        return self._rv(res, names, column_types, lineage)

    def remove_column_in_place(self, name):
        """
        Remove a column from the RDD.

        This operation modifies the current XFrame in place and returns self.
        """
        self._entry(name=name)
        col = self.col_names.index(name)
        self.col_names.pop(col)
        self.column_types.pop(col)

        def pop_col(row):
            lst = list(row)
            lst.pop(col)
            return tuple(lst)
        res = self._dstream.map(pop_col)
        lineage = self.lineage.remove_column(name)
        return self._replace(res, lineage=lineage)

    def remove_columns(self, column_names):
        """
        Remove columns from the RDD.

        This operation creates a new XStreamImpl and returns it.
        """
        self._entry(col_names=column_names)
        cols = [self.col_names.index(name) for name in column_names]
        # pop from highest to lowest does not foul up indexes
        cols.sort(reverse=True)
        remaining_col_names = copy.copy(self.col_names)
        remaining_col_types = copy.copy(self.column_types)
        for col in cols:
            remaining_col_names.pop(col)
            remaining_col_types.pop(col)

        def pop_cols(row):
            lst = list(row)
            for col in cols:
                lst.pop(col)
            return tuple(lst)
        res = self._dstream.map(pop_cols)
        lineage = self.lineage.remove_columns(column_names)
        return self._rv(res, remaining_col_names, remaining_col_types, lineage)

    def swap_columns(self, column_1, column_2):
        """
        Creates an RDD with the given columns swapped.

        This operation
        """
        self._entry(column_1=column_1, column_2=column_2)

        col1 = self.col_names.index(column_1)
        col2 = self.col_names.index(column_2)

        def swap_list(lst):
            new_list = list(lst)
            new_list[col1] = lst[col2]
            new_list[col2] = lst[col1]
            return new_list

        def swap_cols(row):
            # is it OK to modify
            # the row ?
            try:
                lst = list(row)
                lst[col1], lst[col2] = lst[col2], lst[col1]
                return tuple(lst)
            except IndexError:
                logging.warn('Swap index error {} {} {} {}'.format(col1, col2, row, len(row)))
        names = swap_list(self.col_names)
        types = swap_list(self.column_types)
        res = self._dstream.map(swap_cols)
        return self._rv(res, names, types)

    def reorder_columns(self, column_names):
        """
        Return new XStreamImpl, with columns reordered.
        """
        self._entry(column_names=column_names)
        column_indexes = [self.col_names.index(col) for col in column_names]

        def reorder_list(lst):
            return [lst[i] for i in column_indexes]

        def reorder_cols(row):
            return tuple([row[i] for i in column_indexes])

        names = reorder_list(self.col_names)
        types = reorder_list(self.column_types)
        res = self._dstream.map(reorder_cols)
        return self._rv(res, names, types)

    def replace_column_names(self, new_names):
        """
        Return new XStreamImpl, with column names replaced.
        """
        self._entry(new_names=new_names)
        name_map = {k: v for k, v in zip(self.col_names, new_names)}
        lineage = self.lineage.replace_column_names(name_map)
        return self._rv(self._dstream, new_names, lineage=lineage)

    def add_column_const_in_place(self, name, value):
        """
        Add a new column at the end of the DStream with a const value.

        This operation modifies the current DStream in place and returns self.
        """
        self._entry(name=name, value=value)

        def add_col(row):
            row = list(row)
            row.append(value)
            return tuple(row)
        res = self._dstream.map(add_col)

        self.col_names.append(name)
        col_type = type(value)
        self.column_types.append(col_type)
        lineage = self.lineage.add_column_const(name)
        return self._replace(res, lineage=lineage)

    def replace_column_const_in_place(self, name, value):
        """
        Replace thge given column of the DStream with a const value.

        This operation modifies the current DStream in place and returns self.
        """
        self._entry(name=name, value=value)
        index = self.col_names.index(name)

        def replace_col(row):
            row = list(row)
            row[index] = value
            return tuple(row)
        res = self._dstream.map(replace_col)

        column_type = type(value)
        self.column_types[index] = column_type
        lineage = self.lineage.add_column_const(name)
        return self._replace(res, lineage=lineage)

    def replace_single_column_in_place(self, column_name, col, column_type):
        """
        Replace the column in a single-column table with the given one.

        This operation modifies the current XStream in place and returns self.
        """
        self._entry()
        res = col.rdd().map(lambda item: (item, ))
        self.column_types[0] = column_type
        lineage = self.lineage.replace_column(col, column_name)
        return self._replace(res, lineage=lineage)

    def replace_selected_column(self, column_name, col, column_type):
        """
        Replace the given column  with the given one.

        This operation returns a new XStream.
        """
        self._entry(column_name=column_name)
        rdd = self._dstream.zip(col.rdd())
        index = self.col_names.index(column_name)

        def replace_col(row_col):
            row = list(row_col[0])
            col = row_col[1]
            row[index] = col
            return tuple(row)
        res = rdd.map(replace_col)
        names = copy.copy(self.column_names())
        names[index] = column_name
        column_types = copy.copy(self.column_types)
        column_types[index] = column_type
        lineage = self.lineage.replace_column(col, column_name)
        return self._rv(res, names, column_types, lineage)

    def replace_selected_column_in_place(self, column_name, column_type, col):
        """
        Replace the given column  with the given one.

        This operation modifies the current XFrame in place and returns self.
        """
        self._entry(column_name=column_name)
        rdd = self._dstream.zip(col.rdd())
        index = self.col_names.index(column_name)

        def replace_col(row_col):
            row = list(row_col[0])
            col = row_col[1]
            row[index] = col
            return tuple(row)
        res = rdd.map(replace_col)
        self.column_types[index] = column_type
        lineage = self.lineage.replace_column(col, column_name)
        return self._replace(res, lineage=lineage)

    # noinspection PyMethodMayBeStatic
    def groupby_aggregate(self, key_columns_array, group_columns, group_output_columns, group_properties):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def copy_range(self, start, step, stop):
        raise UnimplementedException()

    # noinspection PyMethodMayBeStatic
    def join(self, right, how, join_keys):
        raise UnimplementedException()

    #####################
    #  Output Operations
    #####################

    def process_rows(self, row_fn, init_fn, final_fn):
        self._entry()
        names = self.col_names

        def process_rdd_rows(rdd):
            xf = XFrame.from_rdd(rdd, column_names=names)
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
        column_names = self.column_names()

        def print_rdd_rows(rdd):
            xf = XFrame.from_rdd(rdd, column_names=column_names)
            xf.print_rows(num_rows, num_columns, max_column_width, max_row_width,
                          wrap_text, max_wrap_rows, footer)

        self._dstream.foreachRDD(print_rdd_rows)

    def print_count(self):
        def print_rdd(rdd):
            print rdd.count()
        self._dstream.foreachRDD(print_rdd)
