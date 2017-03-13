"""
This module defines the XStream class which provides the
ability to process streaming operations.
"""

import array
import inspect
import types
import copy

from xframes.xstream_impl import XStreamImpl
from xframes.xframe import XFrame
from xframes.xarray import XArray
from xframes import object_utils

"""
Copyright (c) 2017, Charles Hayden
All rights reserved.
"""

__all__ = ['XStream']


class XStream(object):
    """
    Provides for streams of XFrames.

    An XStream represents a time sequence of XFrames.  These are usually read from a
    live sources, and are processed in batches at a selectable interval.

    XStream objects encapsulate the logic associated with the stream.
    The interface includes a number of class methods that act as factory methods,
    connecting up to external systems are returning an XStream.

    XStream also includes a number of transformers, taking one or two XStreams and transforming them
    into another XStream.

    Finally, XStream includes a number of sinks, that print, save, or expose the stream to external systems.

    XFrame instances are created immediately (and can be used in Jupyter notebooks without restrictions).
    But data does not flow through the streams until the application calls "start".  This data flow happens in
    another thread, so your program gets control back immediately after calling "start".

    Methods that print data (such as print_frames) do not produce output until data starts flowing.  Their output
    goes to stdout, along with anything that you main thread is doing, which works well in a notebook environment.

    As with the other parts of XFrames (and Spark) many of the operators take functional arguments, containing
    the actions to be applied to the data structures.  These functions run in a worker environment, not on
    the main thread (they run in another process, generally on another machine).  Thus you will not see anythin
    that you write to stdout or stderr from these functions.  If you know where to look, you can find this output in the
    Spark worker log files.
    """

    def __init__(self, impl=None, verbose=False):
        """
        Construct an XStream.  You rarely construct an XStream directly, but through the factory methods.

        Parameters
        ----------

        verbose : bool, optional
            If True, print progress messages.

        See Also
        --------
        xframes.XStream.create_from_text_files
            Create an XStream from text files.

        xframes.XStream.create_from_socket_stream
            Create an XStream from data received over a socket.

        xframes.XStream.create_from_kafka_topic
            Create from a kafka topic.
        """
        self._verbose = verbose
        if impl:
            self._impl = impl
            return

        self._impl = XStreamImpl(None)

    def num_columns(self):
        """
        The number of columns in this XFrame.

        Returns
        -------
        int
            Number of columns in the XFrame.

        See Also
        --------
        xframes.XFrame.num_rows
            Returns the number of rows.
        """
        return self._impl.num_columns()

    @staticmethod
    def set_checkpoint(checkpoint_dir):
        """
        Set the checkpoint director for storing state.

        Parameters
        ----------
        checkpoint_dir : string
            Path to a directory for storing checkpoints
        """
        XStreamImpl.set_checkpoint(checkpoint_dir)

    @staticmethod
    def create_from_text_files(directory_path):
        """
        Create XStream (stream of XFrames) from text gathered files in a directory.

        Monitors the directory.  As new files are added, they are read into XFrames and
        introduced to the stream.

        Parameters
        ----------
        directory_path : str
            The directory where files are stored.

        Returns
        -------
        XStream
            An XStream (of XFrames) made up or rows read from files in the directory.
        """
        impl = XStreamImpl.create_from_text_files(directory_path)
        return XStream(impl=impl)

    @staticmethod
    def create_from_socket_stream(hostname, port):
        """
        Create XStream (stream of XFrames) from text gathered from a socket.

        Parameters
        ----------
        hostname : str
            The data hostname.

        port : str
            The port to connect to.

        Returns
        -------
        XStream
            An XStream (of XFrames) made up or rows read from the socket.
        """
        impl = XStreamImpl.create_from_socket_stream(hostname, port)
        return XStream(impl=impl)

    @staticmethod
    def create_from_kafka_topic(topics, kafka_servers=None, kafka_params=None):
        """
        Create XStream (stream of XFrames) from one or more kafka topics.

        Records will be read from a kafka topic or topics.  Each read delivers a group of messages,
        as controlled by the consumer params.  These records are converted into an XFrame using the
        ingest function, and are processed sequentially.

        Parameters
        ----------
        topics : string | list
            A single topic name, or a list of topic names.  These are kafka topics that are
            used to get data from kafka.

        kafka_servers : string | list, optional
            A single kafka server or a list of kafka servers.  Each server is of the form server-name:port.
            If no server is given, the server "localhost:9002" is used.

        kafka_params : dict, optional
            A dictionary of param name - value pairs.  These are passed to kafka as consumer
            configuration parameters..
            See kafka documentation
            http://kafka.apache.org/documentation.html#newconsumerconfigs for
            more details on kafka consumer configuration params.
            If no kafka params are supplied, the list of kafka servers specified in this
            function is passed as the "bootstrap.servers" param.
        """
        if isinstance(topics, basestring):
            topics = [topics]
        if not isinstance(topics, list):
            raise TypeError('Topics must be string or list.')
        if kafka_servers is None:
            kafka_servers = 'localhost:9092'
        elif isinstance(topics, list):
            kafka_servers = ','.join(kafka_servers)

        impl = XStreamImpl.create_from_kafka_topic(topics, kafka_servers, kafka_params)
        return XStream(impl=impl)

    @staticmethod
    def start():
        """
        Start the streaming pipeline running.

        It will continue to run, processing XFrames, until stopped.
        """
        XStreamImpl.start()

    @staticmethod
    def stop(stop_spark_context=True, stop_gracefully=False):
        """
        Stop the streaming pipeline.

        Parameters
        ----------
        stop_spark_context : boolean, optional
            If True, also stop the streaming context.  This releases resources, but it can not be
            started again.  If False, then streaming may be started again.
            Defaults to True.

        stop_gracefully : boolean, optional
            If True, stops gracefully by letting all operations in progress finish before stopping.
            Defaults to false.
        """
        XStreamImpl.stop(stop_spark_context, stop_gracefully)

    @staticmethod
    def await_termination(timeout=None):
        """
        Wait for streaming execution to stop.

        Parameters
        ----------
        timeout : int, optional
            The maximum time to wait, in seconds.
            If not given, wait indefinitely.

        Returns
        -------
        status : boolean
            True if the stream has stopped.  False if the given timeout has expired and the timeout expired.
        """
        return XStreamImpl.await_termination(timeout)

    def impl(self):
        return self._impl

    def dump_debug_info(self):
        """
        Print information about the Spark RDD associated with this XFrame.

        See Also
        --------
        xframes.XFrame.dump_debug_info
            Corresponding function on individual frame.

        """
        return self._impl.dump_debug_info()

    def column_names(self):
        """
        The name of each column in the XStream.

        Returns
        -------
        list[string]
            Column names of the XStream.

        See Also
        --------
        xframes.XFrame.column_names
            Corresponding function on individual frame.

        """
        return copy.copy(self._impl.column_names())

    def column_types(self):
        """
        The type of each column in the XFrame.

        Returns
        -------
        list[type]
            Column types of the XFrame.

        See Also
        --------
        xframes.XFrame.column_types
            Corresponding function on individual frame.
        """
        return copy.copy(self._impl.dtype())

    def dtype(self):
        """
        The type of each column in the XFrame.

        Returns
        -------
        list[type]
            Column types of the XFrame.

        See Also
        --------
        xframes.XFrame.dtype
            Corresponding function on individual frame.

        """
        return self.column_types()

    def lineage(self):
        """
        The table lineage: the files that went into building this table.

        Returns
        -------
        dict
            * key 'table': set[filename]
                The files that were used to build the XArray
            * key 'column': dict{col_name: set[filename]}
                The set of files that were used to build each column

        See Also
        --------
        xframes.XFrame.lineage
            Corresponding function on individual frame.

        """
        return self._impl.lineage_as_dict()

    def num_rows(self):
        """
        Counts the rows in each XFrame in the stream.

        Returns
        -------
        stream of XFrames
            Returns a new XStream consisting of one-row XFrames.
            Each XFrame has one column, "count" containing the number of
            rows in each consittuent XFrame.

        See Also
        --------
        xframes.XFrame.num_rows
            Corresponding function on individual frame.

        """
        return XStream(impl=self._impl.num_rows())

    def to_dstream(self):
        """
        Convert the current XStream to a Spark DStream.  The RDD contained in the DStream
        consists of tuples containing the column data.  No conversion is necessary: the internal DStream is
        returned.

        Returns
        -------
        spark.DStream
            The spark DStream that is used to represent the XStream.

        See Also
        --------
        xframes.XFrame.to_rdd
            Converts to a Spark RDD.
            Corresponding function on individual frame.

        """
        return self._impl.to_dstream()

    @classmethod
    def from_dstream(cls, dstream, col_names, column_types):
        """
        Create a XStream from a spark DStream.  The data should be:

        Parameters
        ----------
        dstream : spark.DStream
            Data used to populate the XStream

        col_names : list of string
            The column names to use.

        column_types : list of type
            The column types to use.

        Returns
        -------
        :class:`.XStream`

        See Also
        --------
        from_rdd
            Converts from a Spark RDD.
            Corresponding function on individual frame.
        """
        return XStream(impl=XStreamImpl(dstream, col_names, column_types))

    def _groupby(self, key_columns, operations, *args):

        # TODO: groupby CONCAT produces unicode output from utf8 input
        # TODO: Preserve character encoding.

        operations = operations or {}
        # some basic checking first
        # make sure key_columns is a list
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        # check that every column is a string, and is a valid column name
        my_column_names = self.column_names()
        my_column_types = self.column_types()
        key_columns_array = []
        for column in key_columns:
            if not isinstance(column, str):
                raise TypeError('Column name must be a string.')
            if column not in my_column_names:
                raise KeyError("Column '{}' does not exist in XFrame".format(column))
            col_type = my_column_types[my_column_names.index(column)]
            if col_type is dict:
                raise TypeError('Cannot group on a dictionary column.')
            key_columns_array.append(column)

        group_output_columns = []
        group_columns = []
        group_properties = []

        all_ops = [operations] + list(args)
        for op_entry in all_ops:
            # if it is not a dict, nor a list, it is just a single aggregator
            # element (probably COUNT). wrap it in a list so we can reuse the
            # list processing code
            operation = op_entry
            if not (isinstance(operation, list) or isinstance(operation, dict)):
                operation = [operation]
            if isinstance(operation, dict):
                # now sweep the dict and add to group_columns and group_properties
                for key, val in operation.iteritems():
                    if not isinstance(val, tuple) and not callable(val):
                        raise TypeError("Unexpected type in aggregator definition of output column: '{}'"
                                        .format(key))
                    if callable(val):
                        prop, column = val()
                    else:
                        prop, column = val
                    num_args = prop.num_args
                    if num_args == 2 and (isinstance(column[0], tuple)) != (isinstance(key, tuple)):
                        raise TypeError('Output column(s) and aggregate column(s) for ' +
                                        'aggregate operation should be either all tuple or all string.')

                    if num_args == 2 and isinstance(column[0], tuple):
                        for (col, output) in zip(column[0], key):
                            group_columns += [[col, column[1]]]
                            group_properties += [prop]
                            group_output_columns += [output]
                    else:
                        group_columns += [column]
                        group_properties += [prop]
                        group_output_columns += [key]

            elif isinstance(operation, list):
                # we will be using automatically defined column names
                for val in operation:
                    if not isinstance(val, tuple) and not callable(val):
                        raise TypeError('Unexpected type in aggregator definition.')
                    if callable(val):
                        prop, column = val()
                    else:
                        prop, column = val
                    num_args = prop.num_args
                    if num_args == 2 and isinstance(column[0], tuple):
                        for col in column[0]:
                            group_columns += [[col, column[1]]]
                            group_properties += [prop]
                            group_output_columns += ['']
                    else:
                        group_columns += [column]
                        group_properties += [prop]
                        group_output_columns += ['']

        # let's validate group_columns
        for cols in group_columns:
            for col in cols:
                if not isinstance(col, str):
                    raise TypeError('Column name must be a string.')

                # TODO:  test for num_args != 0 or don't store empty column name
                if col != '' and col not in my_column_names:
                    raise KeyError("Column '{}' does not exist in XFrame.".format(col))

        return XFrame(impl=self._impl.groupby_aggregate(key_columns_array,
                                                        group_columns,
                                                        group_output_columns,
                                                        group_properties))

    def groupby(self, key_columns, operations=None, *args):
        """
        Perform a group on the `key_columns` followed by aggregations on the
        columns listed in `operations`.

        The `operations` parameter is a dictionary that indicates which
        aggregation operators to use and which columns to use them on. The
        available operators are SUM, MAX, MIN, COUNT, MEAN, VARIANCE, STD, CONCAT,
        SELECT_ONE, ARGMIN, ARGMAX, and QUANTILE.
        See :mod:`~xframes.aggregate` for more detail on the aggregators.

        Parameters
        ----------
        key_columns : string | list[string]
            Column(s) to group by. Key columns can be of any type other than
            dictionary.

        operations : dict, list, optional
            Dictionary of columns and aggregation operations. Each key is a
            output column name and each value is an aggregator. This can also
            be a list of aggregators, in which case column names will be
            automatically assigned.

        \*args
            All other remaining arguments will be interpreted in the same
            way as the operations argument.

        Returns
        -------
        out_xf : XFrame
            A new XFrame, with a column for each groupby column and each
            aggregation operation.

        See Also
        --------
        xframes.XFrame.groupby
            Corresponding function on individual frame.

        """
        return self._groupby(key_columns, operations, *args)

    def count_distinct(self, col):
        """
        Counts the number of different values in a column of each XFrame in the stream.

        Returns
        -------
        stream of XFrames
            Returns a new XStream consisting of one-row XFrames.
            Each XFrame has one column, "count" containing the number of
            rows in each consittuent XFrame.

        """
        names = self._impl.column_names()
        if col not in names:
            raise ValueError('Column name must be in XStream')
        return XStream(impl=self._impl.count_distinct(col))

    def flat_map(self, column_names, fn, column_types='auto'):
        """
        Map each row of each XFrame to multiple rows in a new XFrame via a
        function.

        The output of `fn` must have type ``list[list[...]]``.  Each inner list
        will be a single row in the new output, and the collection of these
        rows within the outer list make up the data for the output XFrame.
        All rows must have the same length and the same order of types to
        make sure the result columns are homogeneously typed.  For example, if
        the first element emitted into the outer list by `fn` is
        ``[43, 2.3, 'string']``, then all other elements emitted into the outer
        list must be a list with three elements, where the first is an `int`,
        second is a `float`, and third is a `string`.  If `column_types` is not
        specified, the first 10 rows of the XFrame are used to determine the
        column types of the returned XFrame.

        Parameters
        ----------
        column_names : list[str]
            The column names for the returned XFrame.

        fn : function
            The function that maps each of the xframe rows into multiple rows,
            returning ``list[list[...]]``.  All output rows must have the same
            length and order of types.  The function is passed a dictionary
            of column name: value for each row.

        column_types : list[type]
            The column types of the output XFrame.

        Returns
        -------
        :class:`.XStream`
            A new XStream containing the results of the ``flat_map`` of the
            XFrames in the XStream.

        See Also
        --------
        xframes.XFrame.flat_map
            Corresponding function on individual frame.
        """
        if not inspect.isfunction(fn):
            raise TypeError('Input must be a function')

        # determine the column_types
        if not isinstance(column_types, list):
            raise TypeError('Column_types must be a list: {} {}.'.format(type(column_types).__name__, column_types))
        if not len(column_types) == len(column_names):
            raise ValueError('Number of output columns must match the size of column names.')
        return XStream(impl=self._impl.flat_map(fn, column_names, column_types))

    def transform_row(self, row_fn, column_names, column_types):
        column_names = column_names
        column_types = column_types
        if len(column_names) != len(column_types):
            raise ValueError('Column_names must be same length as column_types: {} {}'.
                             format(len(column_names), len(column_types)))
        if not inspect.isfunction(row_fn):
            raise TypeError('Row_fn must be a function.')
        return XStream(impl=self._impl.transform_row(row_fn, column_names, column_types))

    def apply(self, fn, dtype):
        """
        Transform each XFrame in an XStream to an :class:`~xframes.XArray` according to a
        specified function. Returns a XStream of XArray of `dtype` where each element
        in this XArray is transformed by `fn(x)` where `x` is a single row in
        the xframe represented as a dictionary.  The `fn` should return
        exactly one value which can be cast into type `dtype`.

        Parameters
        ----------
        fn : function
            The function to transform each row of the XFrame. The return
            type should be convertible to `dtype` if `dtype` is not None.

        dtype : data type
            The `dtype` of the new XArray. If None, the first 100
            elements of the array are used to guess the target
            data type.

        Returns
        -------
        :class:`.XStream`
            The stream of XArray transformed by fn.  Each element of the XArray is of
            type `dtype`
        """
        if not inspect.isfunction(fn):
            raise TypeError('Input must be a function.')
        if not type(dtype) is type:
            raise TypeError('Dtype must be a type')

        return XStream(impl=self._impl.apply(fn, dtype))

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
        :class:`.XStream`
            An XStream with the given column transformed by the function and cast to the given type.
        """
        names = self._impl.column_names()
        if col not in names:
            raise ValueError('Column name must be in XStream')
        if fn is None:
            def fn(row):
                return row[col]
        elif not inspect.isfunction(fn):
            raise TypeError('Fn must be a function.')
        if type(dtype) is not type:
            raise TypeError('Dtype must be a type.')

        return XStream(impl=self._impl.transform_col(col, fn, dtype))

    def filterby(self, values, col_name, exclude=False):
        """
        Filter an XStream by values inside an iterable object. Result is an
        XStream that only includes (or excludes) the rows that have a column
        with the given `column_name` which holds one of the values in the
        given `values` :class:`~xframes.XArray`. If `values` is not an
        XArray, we attempt to convert it to one before filtering.

        Parameters
        ----------
        values : XArray | list |tuple | set | iterable | numpy.ndarray | pandas.Series | str | function
            The values to use to filter the XFrame.  The resulting XFrame will
            only include rows that have one of these values in the given
            column.
            If this is f function, it is called on each row and is passed the value in the
            column given by 'column_name'.  The result includes
            rows where the function returns True.

        col_name : str | None
            The column of the XFrame to match with the given `values`.  This can only be None if the values
            argument is a function.  In this case, the function is passed the whole row.

        exclude : bool
            If True, the result XFrame will contain all rows EXCEPT those that
            have one of `values` in `column_name`.

        Returns
        -------
        :class:`.XStream`
            The filtered XStream.

        See Also
        --------
        xframes.XFrame.filterby
            Corresponding function on individual frame.
        """
        if isinstance(values, types.FunctionType) and col_name is None:
            return XStream(impl=self._impl.filter_by_function_row(values, exclude))

        if not isinstance(col_name, str):
            raise TypeError('Column_name must be a string.')

        existing_columns = self.column_names()
        if col_name not in existing_columns:
            raise KeyError("Column '{}' not in XFrame.".format(col_name))

        if isinstance(values, types.FunctionType):
            return XStream(impl=self._impl.filter_by_function(values, col_name, exclude))

        existing_type = self.column_types()[existing_columns.index(col_name)]

        # If we are given the values directly, use filter.
        if not isinstance(values, XArray):
            # If we were given a single element, put into a set.
            # If iterable, then convert to a set.

            if isinstance(values, basestring):
                # Strings are iterable, but we don't want a set of characters.
                values = {values}
            elif not hasattr(values, '__iter__'):
                values = {values}
            else:
                # Make a new set from the iterable.
                values = set(values)

            if len(values) == 0:
                raise ValueError('Value list is empty.')

            value_type = type(next(iter(values)))
            if value_type != existing_type:
                raise TypeError("Value type ({}) does not match column type ({}).".format(
                    value_type.__name__, existing_type.__name__))
            return XStream(impl=self._impl.filter(values, col_name, exclude))

        # TODO: Below is unimplemented
        # If we have xArray, then use a different strategy based on join.
        value_xf = XStream().add_column(values, col_name)

        # Make sure the values list has unique values, or else join will not filter.
        value_xf = value_xf.groupby(col_name, {})

        existing_type = self.column_types()[existing_columns.index(col_name)]
        given_type = value_xf.column_types()[0]
        if given_type is not existing_type:
            raise TypeError("Type of given values ('{}') does not match type of column '{}' ('{}') in XFrame."
                            .format(given_type, col_name, existing_type))

        if exclude:
            id_name = "id"
            # Make sure this name is unique so we know what to remove in
            # the result
            while id_name in existing_columns:
                id_name += '1'
            value_xf = value_xf.add_row_number(id_name)

            tmp = XStream(impl=self._impl.join(value_xf.impl(),
                                               'left',
                                               {col_name: col_name}))
            # DO NOT CHANGE the next line -- it is xArray operator
            ret_xf = tmp[tmp[id_name] == None]
            del ret_xf[id_name]
            return ret_xf
        else:
            return XStream(impl=self._impl.join(value_xf.impl(),
                                                'inner',
                                                {col_name: col_name}))

    def select_column(self, column_name):
        """
        Return an XStream of :class:`~xframes.XArray` that corresponds with
        the given column name. Throws an exception if the column name is something other than a
        string or if the column name is not found.

        Subscripting an XStream by a column name is equivalent to this function.

        Parameters
        ----------
        column_name : str
            The column name.

        Returns
        -------
        :class:`.XStream`
            The XStream of XArray that is referred by `column_name`.

        See Also
        --------
        xframes.XFrame.select_column
            Corresponding function on individual frame.
        """
        if not isinstance(column_name, str):
            raise TypeError('Invalid column_name type: must be str.')
        return XStream(impl=self._impl.select_column(column_name))

    def select_columns(self, keylist):
        """
        Get XFrame composed only of the columns referred to in the given list of
        keys. Throws an exception if ANY of the keys are not in this XFrame or
        if `keylist` is anything other than a list of strings.

        Parameters
        ----------
        keylist : list[str]
            The list of column names.

        Returns
        -------
        :class:`.XFrame`
            A new XFrame that is made up of the columns referred to in
            `keylist` from the current XFrame.  The order of the columns
            is preserved.

        See Also
        --------
        xframes.XFrame.select_columns
            Corresponding function on individual frame.
        """
        if not hasattr(keylist, '__iter__'):
            raise TypeError('Keylist must be an iterable.')
        if not all([isinstance(x, str) for x in keylist]):
            raise TypeError('Invalid key type: must be str.')

        key_set = set(keylist)
        if len(key_set) != len(keylist):
            for key in key_set:
                if keylist.count(key) > 1:
                    raise ValueError("There are duplicate keys in key list: '{}'.".format(key))

        return XStream(impl=self._impl.select_columns(keylist))

    def add_column(self, col, name=''):
        """
        Add a column to every XFrame in this XStream. The length of the new column
        must match the length of the existing XFrame. This
        operation returns new XFrames with the additional columns.
        If no `name` is given, a default name is chosen.

        Parameters
        ----------
        col : XArray
            The 'column' of data to add.

        name : string, optional
            The name of the column. If no name is given, a default name is
            chosen.

        Returns
        -------
        :class:`.XStream`
            A new XStream of XFrame with the new column.

        See Also
        --------
        xframes.XFrame.add_column
            Corresponding function on individual frame.
        """
        # Check type for pandas dataframe or XArray?
        if not isinstance(col, XArray):
            raise TypeError('Must give column as XArray.')
        if not isinstance(name, str):
            raise TypeError('Invalid column name: must be str.')
        return XStream(impl=self._impl.add_column(col.impl(), name))

    def add_columns(self, cols, namelist=None):
        """
        Adds multiple columns to this XFrame. The length of the new columns
        must match the length of the existing XFrame. This
        operation returns a new XFrame with the additional columns.

        Parameters
        ----------
        cols : list of XArray or XFrame
            The columns to add.  If `cols` is an XFrame, all columns in it are added.

        namelist : list of string, optional
            A list of column names. All names must be specified. `Namelist` is
            ignored if `cols` is an XFrame.  If there are columns with duplicate names, they
            will be made unambiguous by adding .1 to the second copy.

        Returns
        -------
        :class:`.XFrame`
            The XFrame with additional columns.

        See Also
        --------
        xframes.XFrame.add_columns
            Corresponding function on individual frame.
        """
        if isinstance(cols, XFrame):
            return XStream(impl=self._impl.add_columns_frame(cols.impl()))
        else:
            if not hasattr(cols, '__iter__'):
                raise TypeError('Column list must be an iterable.')
            if not hasattr(namelist, '__iter__'):
                raise TypeError('Namelist must be an iterable.')

            if not all([isinstance(x, XArray) for x in cols]):
                raise TypeError('Must give column as XArray.')
            if not all([isinstance(x, str) for x in namelist]):
                raise TypeError("Invalid column name in list : must all be 'str'.")
            if len(namelist) != len(cols):
                raise ValueError('Namelist length mismatch.')

            cols_impl = [col.impl() for col in cols]
            return XStream(impl=self._impl.add_columns_array(cols_impl, namelist))

    def replace_column(self, name, col):
        """
        Replace a column in this XFrame. The length of the new column
        must match the length of the existing XFrame. This
        operation returns a new XFrame with the replacement column.

        Parameters
        ----------
        name : string
            The name of the column.

        col : XArray
            The 'column' to add.

        Returns
        -------
        :class:`.XFrame`
            A new XFrame with specified column replaced.

        See Also
        --------
        xframes.XFrame.replace_column
            Corresponding function on individual frame.
        """
        # Check type for pandas dataframe or XArray?
        if not isinstance(col, XArray):
            raise TypeError('Must give column as XArray.')
        if not isinstance(name, str):
            raise TypeError('Invalid column name: must be str.')
        if name not in self.column_names():
            raise ValueError('Column name must be in XFrame')
        return XStream(impl=self._impl.replace_selected_column(name, col.impl(), col.dtype()))

    def remove_column(self, name):
        """
        Remove a column from this XFrame. This
        operation returns a new XFrame with the given column removed.

        Parameters
        ----------
        name : string
            The name of the column to remove.

        Returns
        -------
        :class:`.XFrame`
            A new XFrame with given column removed.

        See Also
        --------
        xframes.XFrame.remove_column
            Corresponding function on individual frame.
        """
        if name not in self.column_names():
            raise KeyError('Cannot find column {}.'.format(name))
        return XStream(impl=self._impl.remove_column(name))

    def remove_columns(self, column_names):
        """
        Removes one or more columns from this XFrame. This
        operation returns a new XFrame with the given columns removed.

        Parameters
        ----------
        column_names : list or iterable
            A list or iterable of the column names.

        Returns
        -------
        :class:`.XFrame`
            A new XFrame with given columns removed.

        See Also
        --------
        xframes.XFrame.remove_columns
            Corresponding function on individual frame.
        """
        if not hasattr(column_names, '__iter__'):
            raise TypeError('Column_names must be an iterable.')
        for name in column_names:
            if name not in self.column_names():
                raise KeyError('Cannot find column {}.'.format(name))
        return XStream(impl=self._impl.remove_columns(column_names))

    def swap_columns(self, column_1, column_2):
        """
        Swap the columns with the given names. This
        operation returns a new XFrame with the given columns swapped.

        Parameters
        ----------
        column_1 : string
            Name of column to swap

        column_2 : string
            Name of other column to swap

        Returns
        -------
        :class:`.XFrame`
            A new XFrame with specified columns swapped.

        See Also
        --------
        xframes.XFrame.swap_columns
            Corresponding function on individual frame.
        """
        if column_1 not in self.column_names():
            raise KeyError("Cannot find column '{}'.".format(column_1))
        if column_2 not in self.column_names():
            raise KeyError("Cannot find column '{}'.".format(column_2))

        return XStream(impl=self._impl.swap_columns(column_1, column_2))

    def reorder_columns(self, column_names):
        """
        Reorder the columns in the table.  This
        operation returns a new XFrame with the given columns reordered.

        Parameters
        ----------
        column_names : list of string
            Names of the columns in desired order.

        Returns
        -------
        :class:`.XFrame`
            A new XFrame with reordered columns.

        See Also
        --------
        xframes.XFrame.reorder_columns
            Corresponding function on individual frame.
        """
        if not hasattr(column_names, '__iter__'):
            raise TypeError('Keylist must be an iterable.')
        for col in column_names:
            if col not in self.column_names():
                raise KeyError("Cannot find column '{}'.".format(col))
        for col in self.column_names():
            if col not in column_names:
                raise KeyError("Column '{}' not assigned'.".format(col))

        return XStream(impl=self._impl.reorder_columns(column_names))

    def rename(self, names):
        """
        Rename the given columns. `Names` can be a dict specifying
        the old and new names. This changes the names of the columns given as
        the keys and replaces them with the names given as the values.  Alternatively,
        `names` can be a list of the new column names.  In this case it must be
        the same length as the number of columns.  This
        operation returns a new XFrame with the given columns renamed.

        Parameters
        ----------
        names : dict [string, string] | list [ string ]
            Dictionary of [old_name, new_name] or list of new names

        Returns
        -------
        :class:`.XFrame`
            A new XFrame with columns renamed.

        xframes.XFrame.rename
            Corresponding function on individual frame.
        """
        if not isinstance(names, (list, dict)):
            raise TypeError('Names must be a dictionary: oldname -> newname or a list of newname ({}).'
                            .format(type(names).__name__))
        if isinstance(names, dict):
            new_names = copy.copy(self.column_names())
            for k in names:
                if k not in self.column_names():
                    raise ValueError("Cannot find column '{}' in the XFrame.".format(k))
                index = self.column_names().index(k)
                new_names[index] = names[k]
        else:
            new_names = names
            if len(new_names) != len(self.column_names()):
                raise ValueError('Names must be the same length as the number of columns (names: {} columns: {}).'
                                 .format(len(new_names), len(self.column_names())))
        return XStream(impl=self._impl.replace_column_names(new_names))

    def __getitem__(self, key):
        """
        This method does things based on the type of `key`.

        If `key` is:
            * str
              Calls `select_column` on `key`
            * list
              Calls select_columns on `key`
        """
        if isinstance(key, list):
            return self.select_columns(key)
        if isinstance(key, str):
            return self.select_column(key)
        if isinstance(key, unicode):
            return self.select_column(str(key))

        raise TypeError('Invalid index type: must be ' +
                        "''list' or 'str': ({})".format(type(key).__name__))

    def __setitem__(self, key, value):
        """
        Adds columns to each XFrame and returns the modified XStream.

        Key can be either a list or a str.  If
        value is an XArray, it is added to the XFrame as a column.  If it is a
        constant value (int, str, or float), then a column is created where
        every entry is equal to the constant value.  Existing columns can also
        be replaced using this function.

        """
        if isinstance(key, list):
            col_list = value
            if isinstance(value, XFrame):
                for name in value.column_names():
                    if name in self.column_names():
                        raise ValueError("Column '{}' already exists in current XFrame.".format(name))
                self.impl().add_columns_frame_in_place(value.impl())
            else:
                if not hasattr(col_list, '__iter__'):
                    raise TypeError('Column list must be an iterable.')
                if not hasattr(key, '__iter__'):
                    raise TypeError('Namelist must be an iterable.')
                if not all([isinstance(x, XArray) for x in col_list]):
                    raise TypeError('Must give column as XArray.')
                if not all([isinstance(x, str) for x in key]):
                    raise TypeError("Invalid column name in list : must all be 'str'.")
                if len(key) != len(col_list):
                    raise ValueError('Namelist length mismatch.')
                cols_impl = [col.impl() for col in col_list]
                self._impl.add_columns_array_in_place(cols_impl, key)
        elif isinstance(key, str):
            if isinstance(value, XArray):
                sa_value = value
            elif hasattr(value, '__iter__'):  # wrap list, array... to xarray
                sa_value = XArray(value)
            else:
                # Special case of adding a const column.
                # It is very inefficient to create a column and then zip it in
                # a) num_rows() is inefficient
                # b) parallelize is inefficient
                # c) partitions differ, so zip --> zipWithIndex, sortByKey, etc
                # Map it in instead
                if not isinstance(value, (int, float, str, array.array, list, dict)):
                    raise TypeError("Cannot create xarray of value type '{}'.".format(type(value).__name__))
                if key not in self.column_names():
                    self._impl.add_column_const_in_place(key, value)
                else:
                    self._impl.replace_column_const_in_place(key, value)
                return

            # set new column
            if key not in self.column_names():
                self._impl.add_column_in_place(sa_value.impl(), key)
            else:
                # special case if replacing the only column.
                # server would fail the replacement if the new column has different
                # length than current one, which doesn't make sense if we are replacing
                # the only column. To support this, we call a different function in the
                # implementation.
                single_column = (self.num_columns() == 1)
                col_type = self.column_types()[0]
                if single_column:
                    self._impl.replace_single_column_in_place(key, sa_value.impl(), col_type)
                else:
                    self._impl.replace_selected_column_in_place(key, sa_value.impl(), col_type)

        else:
            raise TypeError('Cannot set column with key type {}.'.format(type(key).__name__))

    def __delitem__(self, name):
        """
        Removes a column and returns the modified for each XFram in the XStream.
        """
        if name not in self.column_names():
            raise KeyError('Cannot find column {}.'.format(name))
        self._impl.remove_column_in_place(name)
        return self

    def process_rows(self, row_fn, init_fn=None, final_fn=None):
        """
        Process the rows in a stream of XFrames using a given row processing function.

        This is an output operation, and forces the XFrames to be evaluated.

        Parameters
        ----------
        row_fn : function
            This function is called on each row of each XFrame.
            This function receives two parameters: a row and an initiali value.
            The row is in the form of a dictionary of column-name: column_value pairs.
            The initial value is the return value resulting from calling the init_fn.
            The row_fn need not return a value: the function is called for its side effects only.

        init_fn : function, optional
            The init_fn is a parameterless function, used to set up the environment for the row function.
            Its value is passed to each invocation of the row function.  If no init_fn is passed, then
            each row function will receive None as its second argument.

            The rows are processed in parallel in groups on one or more worker machines.  For each
            group, init_fn is called once, and its return value is passed to each row_fn.  It could be
            used, for instance, to open a file or socket that is used by each of the row functions.

        final_fn : function, optional
            The final_fn is called after each group is processed.  It is a function of one parameter, the
            return value of the initial function.
        """
        self._impl.process_rows(row_fn, init_fn, final_fn)

    def process_frames(self, frame_fn, init_fn=None, final_fn=None):
        # TODO: review iit_fn and final_fn -- how do they work?
        """
        Process the frames in a stream of XFrames using a given frame processing function.

        This is an output operation, and forces the XFrames to be evaluated.

        Parameters
        ----------
        frame_fn : function
            This function is called on each XFrame in the XStream.
            This function receives two parameters: a frame and an initiali value.
            The initial value is the return value resulting from calling the init_fn.
            The frame_fn need not return a value: the function is called for its side effects only.

        init_fn : function, optional
            The init_fn is a parameterless function, used to set up the environment for the frame function.
            Its value is passed to each invocation of the frame function.  If no init_fn is passed, then
            each frame function will receive None as its second argument.

            The rows are processed in parallel in groups on one or more worker machines.  For each
            group, init_fn is called once, and its return value is passed to each row_fn.  It could be
            used, for instance, to open a file or socket that is used by each of the row functions.

        final_fn : function, optional
            The final_fn is called after each group is processed.  It is a function of one parameter, the
            return value of the initial function.
        """
        self._impl.process_frames(frame_fn, init_fn, final_fn)

    def save(self, prefix, suffix=None):
        """
        Save the XStream to a set of files in the file system.

        This is an output operation, and forces the XFrames to be evaluated.

        Parameters
        ----------
        prefix : string
            The base location to save each XFrame in the XStream.
            The filename of each files will be made as follows:
            prefix-TIME-IN-MS.suffix.
            The prefix should be either a local directory or a
            remote URL.
        suffix : string, optional
            The filename suffix.  Defaults to no suffix.

        See Also
        --------
        xframes.XFrame.save
            Corresponding function on individual frame.
        """
        if not isinstance(prefix, basestring):
            raise TypeError('Prefix must be string')
        if suffix is not None and not isinstance(suffix, basestring):
            raise TypeError('Suffix must be string')
        self._impl.save(prefix, suffix)

    def print_frames(self, num_rows=10, num_columns=40,
                     max_column_width=30, max_row_width=None,
                     wrap_text=False, max_wrap_rows=2, footer=False):
        """
        Print the first rows and columns of each XFrame in the XStream in human readable format.

        Parameters
        ----------
        num_rows : int, optional
            Number of rows to print.

        num_columns : int, optional
            Number of columns to print.

        max_column_width : int, optional
            Maximum width of a column. Columns use fewer characters if possible.

        max_row_width : int, optional
            Maximum width of a printed row. Columns beyond this width wrap to a
            new line. `max_row_width` is automatically reset to be the
            larger of itself and `max_column_width`.

        wrap_text : boolean, optional
            Wrap the text within a cell.  Defaults to False.

        max_wrap_rows : int, optional
            When wrapping is in effect, the maximum number of resulting rows for each cell
            before truncation takes place.

        footer : bool, optional
            True to pinrt a footer.

        See Also
        --------
        xframes.XFrame.print_rows
            Corresponding function on individual frame.
        """
        max_row_width = max_row_width or object_utils.MAX_ROW_WIDTH
        self._impl.print_frames(num_rows, num_columns, max_column_width, max_row_width,
                                wrap_text, max_wrap_rows, footer)

    def print_count(self):
        self._impl.print_count()


    def update_state(self, fn, col_name, initial_state):
        # Take the column names and types from the initial state
        """
        Update state for an XStream by using the state key in a given column.

        The state is a key-value store.  The key is made up of the values in the given column.
        For each XFrame in the XStream, all the rows with a given key are passed to the supplied function,
        which computes a new state.

        Parameters
        ----------
        fn : function
            The given function is supplied with a list of rows in each XFrame that have the same value in the given
            column (the key), along with the current state.  It returns the new state for that key.
            The function is: fn(rows, old_state) and returns new_state.

        col_name : str | None
            The column of the XStream to match supplies the state key.

        initial_state : XFrame
            Used to initialize the state.  The update function should return a tuple of values
            that conform to the column names and types of this state.

        Returns
        -------
        An XStream made up of XFrames representing the state.
        """
        if not isinstance(col_name, str):
            raise TypeError('Column_name must be a string.')

        existing_columns = self.column_names()
        if col_name not in existing_columns:
            raise KeyError("Column '{}' not in XFrame.".format(col_name))

        if not isinstance(fn, types.FunctionType):
            raise TypeError('Fn must be a function.')

        if not isinstance(initial_state, XFrame):
            raise TypeError('Initial_state must be an XFrame.')

        return XStream(impl=self._impl.update_state(fn, col_name, initial_state))
