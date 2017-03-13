import inspect

from xframes.xframe import XFrame
from xframes.xstream import XStream
from xframes.xstate_impl import XStateImpl

class XState(object):
    def __init__(self, state, key_column_name, checkpoint_policy, impl=None):
        if impl:
            self._impl = impl
            return

        if not isinstance(state, XFrame):
            raise TypeError("State must be XFrame")
        if key_column_name not in state.column_names():
            raise ValueError('XState key column name is not one of the state columns.')
        if not isinstance(checkpoint_policy, dict):
            raise TypeError('Checkpoint_policy must be a dict.')
        self._impl = XStateImpl(state, key_column_name, checkpoint_policy)


    def update_state(self, fn, stream, column_name):
        if not inspect.isfunction(fn):
            raise TypeError('Fn must be a function.')
        if not isinstance(stream, XStream):
            raise TypeError('Stream is not an XStream.')
        if column_name not in stream.column_names():
            raise ValueError('Column name is not a column in the stream: {} {}.'.
                             format(column_name, stream.column_names()))
        return XStream(impl=self._impl.update_state(fn, stream, column_name))
