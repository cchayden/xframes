"""
This module defines the XStream class which provides the
ability to process streaming operations.
"""

import json
import inspect
import types
import copy


from xframes.xobject import XObject
from xframes.xstream_impl import XStreamImpl
from xframes.xarray import XArray
import xframes

"""
Copyright (c) 2017, Charles Hayden, Inc.
All rights reserved.
"""

__all__ = ['XStream']


class XStream(XObject):
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

        if impl:
            self._impl = impl
            return

        self._impl = XStreamImpl(None)

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
    def create_from_kafka_topics(topics, kafka_servers=None, kafka_params=None):
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

        def default_ingest_fn(kafka_tuple):
            try:
                # loads the kafka value: drop the key and assume it is in json format
                return json.loads(kafka_tuple[1])
            except:
                return None

        impl = XStreamImpl.create_from_kafka_topics(topics, kafka_servers, kafka_params)
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
        out : list[string]
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
        out : list[type]
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
        out : list[type]
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
        out : dict
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
        if not col in names:
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
        out : XStream
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

    def transform_row(self, row_fn, col_names, column_types):
        col_names = col_names or self._impl.column_names()
        column_types = column_types or self._impl.column_types
        if len(col_names) != len(column_types):
            raise ValueError('Col_names must be same length as column_types: {} {}'.\
                             format(len(col_names), len(column_types)))
        if not inspect.isfunction(row_fn):
            raise TypeError('Row_fn must be a function.')
        return XStream(impl=self._impl.transform_row(row_fn, col_names, column_types))

    def transform_col(self, col, fn, dtype):
        names = self._impl.column_names()
        if not col in names:
            raise ValueError('Column name must be in XStream')
        if fn is None:
            def fn(row):
                return row[col]
        elif not inspect.isfunction(fn):
            raise TypeError('Fn must be a function.')
        if type(dtype) is not type:
            raise TypeError('Dtype must be a type.')

        return XStream(impl=self._impl.transform_col(col, fn, dtype))

    def filterby(self, values, column_name, exclude=False):
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

        column_name : str | None
            The column of the XFrame to match with the given `values`.  This can only be None if the values
            argument is a function.  In this case, the function is passed the whole row.

        exclude : bool
            If True, the result XFrame will contain all rows EXCEPT those that
            have one of `values` in `column_name`.

        Returns
        -------
        out : XStream
            The filtered XStream.

        See Also
        --------
        xframes.XFrame.filterby
            Corresponding function on individual frame.
        """
        if isinstance(values, types.FunctionType) and column_name is None:
            return XStream(impl=self._impl.filter_by_function_row(values, exclude))

        if not isinstance(column_name, str):
            raise TypeError('Column_name must be a string.')

        existing_columns = self.column_names()
        if column_name not in existing_columns:
            raise KeyError("Column '{}' not in XFrame.".format(column_name))

        if isinstance(values, types.FunctionType):
            return XStream(impl=self._impl.filter_by_function(values, column_name, exclude))

        existing_type = self.column_types()[existing_columns.index(column_name)]

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
            return XStream(impl=self._impl.filter(values, column_name, exclude))

        # If we have xArray, then use a different strategy based on join.
        value_xf = XStream().add_column(values, column_name)

        # Make sure the values list has unique values, or else join will not filter.
        value_xf = value_xf.groupby(column_name, {})

        existing_type = self.column_types()[existing_columns.index(column_name)]
        given_type = value_xf.column_types()[0]
        if given_type is not existing_type:
            raise TypeError("Type of given values ('{}') does not match type of column '{}' ('{}') in XFrame."
                            .format(given_type, column_name, existing_type))

        if exclude:
            id_name = "id"
            # Make sure this name is unique so we know what to remove in
            # the result
            while id_name in existing_columns:
                id_name += '1'
            value_xf = value_xf.add_row_number(id_name)

            tmp = XStream(impl=self._impl.join(value_xf._impl,
                                              'left',
                                              {column_name: column_name}))
            # DO NOT CHANGE the next line -- it is xArray operator
            ret_xf = tmp[tmp[id_name] == None]
            del ret_xf[id_name]
            return ret_xf
        else:
            return XStream(impl=self._impl.join(value_xf._impl,
                                               'inner',
                                               {column_name: column_name}))

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
                   max_column_width=30, max_row_width=xframes.MAX_ROW_WIDTH,
                   wrap_text=False, max_wrap_rows=2, footer=True):
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
        self._impl.print_frames(num_rows, num_columns, max_column_width, max_row_width,
                         wrap_text, max_wrap_rows, footer)
