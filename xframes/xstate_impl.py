import os

from xframes.traced_object import TracedObject
from xframes.utils import build_row
from xframes.xstream import XStreamImpl
from xframes.spark_context import CommonSparkContext
import xframes.fileio as fileio

class XStateImpl(TracedObject):
    """
    Implements a stateful object used in conjunction with streaming.

    XState holds the state that is maintained as the stream is processed.

    The state is maintained as a structure similar to a frame: a set of rows and
    columns, where each column has a uniform type.  One of the columns is the
    key column.

    For each event in a stream, one of the fields if the event is used as the key
    to identify the state.  This elment of the state is updated for each event
    """

    instance_count = 0

    def __init__(self, state, key_column_name, checkpoint_policy):
        """ Instantiate an XState implementation.

        The RDD holds the state.
        """
        self._entry(checkpoint_policy=checkpoint_policy)
        super(XStateImpl, self).__init__()
        self.checkpoint_policy = checkpoint_policy

        self._rdd = state._impl._rdd
        self.col_names = state.column_names()
        self.column_types = state.column_types()
        self.key_column_name = key_column_name
        self.instance_name = 'state-instance-{}'.format(XStateImpl.instance_count)
        self.checkpoint_interval = checkpoint_policy.get('checkpoint_interval', 10)
        self.checkpoint_policy = checkpoint_policy
        self.checkpoint_path = os.path.join(checkpoint_policy.get('checkpoint_dir', 'checkpoint'), self.instance_name)
        XStateImpl.instance_count += 1

    # TODO this could be housed in XStream and XState instance passed in
    def update_state(self, fn, stream, column_name, num_partitions=None):
        names = stream.column_names()
        index = names.index(column_name)

        state_column_names = self.col_names
        state_column_types = self.column_types

        if num_partitions is None:
            num_partitions = CommonSparkContext.spark_context().defaultParallelism

        def update_fn(events, state):
            """ Calls the user supplied fn, filtering out empty event lists."""
            if len(events) == 0:
                return state
            return fn(events, state)

        # map rows to pair rdds of row-dictionaries
        dstream = stream.impl()._dstream.map(lambda row: (row[index], build_row(names, row)))

        instance_name = self.instance_name
        globals()[instance_name] = self._rdd
        checkpoint_interval = self.checkpoint_interval
        checkpoint_path = self.checkpoint_path
        globals()['checkpoint_counter'] = 0

        #initial_state = None

        def reduce_func(stream_rdd):
            my_state = globals()[instance_name]
            checkpoint_counter = globals()['checkpoint_counter']

            # g represents the value that is passed to the update func: list(rows) and state
            if my_state is None:
                # make empty state
                g = stream_rdd.groupByKey(num_partitions).mapValues(lambda vs: (list(vs), None))
            else:
                # group new items and state under the key
                g = my_state.cogroup(stream_rdd.partitionBy(num_partitions), num_partitions)
                # unravel the results of cogroup into values and state
                # the conditional accounts for state that has no existing value for the key
                g = g.mapValues(lambda state_stream: (list(state_stream[1]), list(state_stream[0])[0]
                    if len(state_stream[0]) else None))

            # apply the user-supplied update function to the values and state
            state = g.mapValues(lambda values_state: update_fn(values_state[0], values_state[1]))
            # remove items from the state that passed back None
            res = state.filter(lambda k_v: k_v[1] is not None)

            checkpoint_counter += 1
            if checkpoint_counter >= checkpoint_interval:
                print 'saving {} {}'.format(checkpoint_counter, checkpoint_path)
                checkpoint_counter = 0
                fileio.delete(checkpoint_path)
                res.saveAsPickleFile(checkpoint_path)
            globals()[instance_name] = res
            globals()['checkpoint_counter'] = checkpoint_counter
            return res

        # Here is where the augmented update function is applied to the RDDs in the stream.
        # This returns the state dstream, in pair-RDD form.
        # TODO we cannot pass an RDD into a transform if checkpointing is on
        # http://stackoverflow.com/questions/30882256/how-to-filter-dstream-using-transform-operation-and-external-rdd
        res = dstream.transform(lambda rdd: reduce_func(rdd))
        # Get the value, throw away the key.
        # TODO: preserve the pair RDD, so the partitioning is stable.
        res = res.map(lambda kv: kv[1])
        # Save a copy of the state for next time, where it becomes the initial_state.
        self._rdd = res
        # Return the state also
        return XStreamImpl(res, column_names=state_column_names, column_types=state_column_types)
