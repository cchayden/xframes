import inspect

from xframes.xframe import XFrame
from xframes.xstream import XStream
from xframes.xstate_impl import XStateImpl

class XState(object):
    """
    Represents state for use with streaming.

    State is able to store information as XFrames are processed in an XStream.
    """
    def __init__(self, initial_state, key_column_name, checkpoint_policy, impl=None):
        """
        Create a new XState object.

        This remembers information as an XStream is processed.

        Parameters
        ----------
        initial_state : XFrame
            This initializes the internal state held in this object.  As XFrames are processed in the XStream,
            the internal state is modified.
            The state's colukn names and types are taken from the initial_state.

        key_column_name : str
            The state is organized by key.  The key is one of the columns of the state.  This
            gives the name of the column, which must be present in the state.

        checkpoint_policy : dict
            The checkpoint policy governs how often and where the state is checkpointed during processing.
        """
        if impl:
            self._impl = impl
            return

        if not isinstance(initial_state, XFrame):
            raise TypeError("State must be XFrame")
        if key_column_name not in initial_state.column_names():
            raise ValueError('XState key column name is not one of the state columns.')
        if not isinstance(checkpoint_policy, dict):
            raise TypeError('Checkpoint_policy must be a dict.')
        self._impl = XStateImpl(initial_state, key_column_name, checkpoint_policy)


    def update_state(self, fn, stream, key_column_name):
        """
        Updates the state as the stream is being processed, by applying a user-supplied function.

        fn : function
            This function is called for each XFrame in the XStream to update the state.
            The update is organized by key: this function is called once for each key in the stream.
            Fn is passed the events (the rows in the stream corresponding the a given key), and the
            portion of the state associated with this key.

        stream : XStream
            The stream supplying the events.

        key_column_name : str
            The name of the column supplying the key.  The stream is made up of a sequence of XFrames,
            each of which has named columns.  This is the name of the column in the *stream* that supplies
            the key.  It may be different that the name of the column in the state.

        Returns
        --------
        XStream
            Returns an XStream made up of the transformed state.  After the update function is applied
            to each group of events, an updated state is produced.  An XStream of these updated states
            is returned when the stream runs.
        """
        if not inspect.isfunction(fn):
            raise TypeError('Fn must be a function.')
        if not isinstance(stream, XStream):
            raise TypeError('Stream is not an XStream.')
        if key_column_name not in stream.column_names():
            raise ValueError('Column name is not a column in the stream: {} {}.'.
                             format(key_column_name, stream.column_names()))
        return XStream(impl=self._impl.update_state(fn, stream, key_column_name))

    def map_with_state(self, fn, stream, key_column_name, column_names, column_types, num_partitions=None):
        if not inspect.isfunction(fn):
            raise TypeError('Fn must be a function.')
        if not isinstance(stream, XStream):
            raise TypeError('Stream is not an XStream.')
        if key_column_name not in stream.column_names():
            raise ValueError('Column name is not a column in the stream: {} {}.'.
                             format(key_column_name, stream.column_names()))
        return XStream(impl=self._impl.map_with_state(fn, stream, key_column_name,
                                                      column_names, column_types,
                                                      num_partitions))

