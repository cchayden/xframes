import os
import time

from xframes.traced_object import TracedObject
from xframes.utils import build_row
from xframes.xstream import XStreamImpl
from xframes.spark_context import CommonSparkContext
import xframes.fileio as fileio

class StateData(object):
    # wraps state so it can be passed by reference and modified in user func

    def __init__(self, state):
        self.state = state

    def get(self):
        return self.state

    def update(self, new_state):
        self.state = new_state


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

    def __init__(self, state, key_column_name, checkpoint_policy, verbose=False):
        """ Instantiate an XState implementation.

        The RDD holds the state.
        """
        self._entry(checkpoint_policy=checkpoint_policy)
        super(XStateImpl, self).__init__()
        self.checkpoint_policy = checkpoint_policy
        self.verbose = verbose

        def checkpoint_filename(path):
            return '{}-checkpoint'.format(path)

        self._rdd = state._impl._rdd
        self.col_names = state.column_names()
        self.column_types = state.column_types()
        self.key_column_name = key_column_name
        self.instance_name = 'state-instance-{}'.format(XStateImpl.instance_count)
        self.checkpoint_interval = checkpoint_policy.get('checkpoint_interval', 10)
        self.checkpoint_policy = checkpoint_policy
        self.checkpoint_path = os.path.join(checkpoint_policy.get('checkpoint_dir', 'checkpoint'),
                                            checkpoint_filename(self.instance_name))

        globals()[self.instance_name] = self._rdd.map(lambda row: (row[key_column_name], row))
        globals()['checkpoint_counter'] = 0
        globals()['batch_counter'] = 0

        XStateImpl.instance_count += 1

    @staticmethod
    def safe_checkpoint(rdd, path):
        temp_path = '{}.temp'.format(path)
        fileio.delete(temp_path)
        rdd.saveAsPickleFile(temp_path)
        fileio.delete(path)
        fileio.rename(temp_path, path)

    ####################
    ### update_state ###
    ####################
    def update_state(self, fn, stream, column_name, num_partitions=None):
        names = stream.column_names()
        index = names.index(column_name)

        checkpoint_interval = self.checkpoint_interval
        checkpoint_path = self.checkpoint_path

        instance_name = self.instance_name

        if num_partitions is None:
            num_partitions = CommonSparkContext.spark_context().defaultParallelism

        def update_fn(events, state):
            """ Calls the user supplied fn, filtering out empty event lists."""
            if len(events) == 0:
                return state
            return fn(events, state)

        # map rows to pair rdds of row-dictionaries
        dstream = stream.impl()._dstream.map(lambda row: (row[index], build_row(names, row)))

        def transform_func(stream_rdd):
            my_state = globals()[instance_name]
            checkpoint_counter = globals()['checkpoint_counter']
            batch_counter = globals()['batch_counter']

            # g represents the value that is passed to the update func: list(rows) and state

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
            batch_counter += 1
            #print 'batch: {}  stream: {}  state: {}'.format(batch_counter, stream_rdd.count(), res.count())
            if checkpoint_counter >= checkpoint_interval:
                if self.verbose:
                    print 'saving {} {}'.format(checkpoint_counter, checkpoint_path)
                checkpoint_counter = 0
                XStateImpl.safe_checkpoint(res, checkpoint_path)
                res = CommonSparkContext().spark_context().pickleFile(checkpoint_path)
            globals()[instance_name] = res
            globals()['checkpoint_counter'] = checkpoint_counter
            globals()['batch_counter'] = batch_counter
            return res

        # Here is where the augmented update function is applied to the RDDs in the stream.
        # This returns the state dstream, in pair-RDD form.
        # TODO we cannot pass an RDD into a transform if checkpointing is on
        # http://stackoverflow.com/questions/30882256/how-to-filter-dstream-using-transform-operation-and-external-rdd
        res = dstream.transform(lambda rdd: transform_func(rdd))
        # Get the value, throw away the key.
        res = res.map(lambda kv: kv[1])
        # Save a copy of the state for next time, where it becomes the initial_state.
        # Return the state also
        state_column_names = self.col_names
        state_column_types = self.column_types
        return XStreamImpl(res, column_names=state_column_names, column_types=state_column_types)

    ####################
    ## map_with_state ##
    ####################

    def map_with_state(self, fn, stream, key_column_name, column_names, column_types, num_partitions):
        """
        Parameters
        ----------
        fn : function
            Takes: key, event, state_data, returns row.
            The state data contains is the state for the key, along with an update function.
        stream : XStream
            Stream of input XFrames
        key_column_name : str
            The name of a column in the stream to use as the key

        Returns
        -------
        DStream
            Stream of XFrames.  Each XFrame contains the rows returned by fn.
            There is one row for each row of the XFrame in the stream.
            The state is updated for each row of the XFrame.
        """
        names = stream.column_names()
        index = names.index(key_column_name)

        checkpoint_interval = self.checkpoint_interval
        checkpoint_path = self.checkpoint_path

        instance_name = self.instance_name

        if num_partitions is None:
            num_partitions = CommonSparkContext.spark_context().defaultParallelism

        def update_fn(key, events, state):
            """
            for each key:
                gets a key, a list of values for the key, and the state for the key
                wrap state
                map over the values for the key
                  passing the wrapped state along
                  collecting the function results
            returning (list of results, final_state)
            """
            if len(events) == 0:
                raise ValueError('update_fn: no events')
            ws = StateData(state)
            if state is not None: print 'old-state: {}'.format(state)
            results = map(lambda event: fn(key, event, ws), events)
            new_state = ws.get()
            #print 'new-state: {}'.format(new_state)
            return results, new_state

        # map rows to pair rdds of row-dictionaries
        dstream = stream.impl()._dstream.map(lambda row: (row[index], build_row(names, row)))

        def transform_func(stream_rdd):
            my_state = globals()[instance_name]
            checkpoint_counter = globals()['checkpoint_counter']
            batch_counter = globals()['batch_counter']

            def merge_state(old_state, new_state):
                def combiner(x):
                    old = list(x[0])
                    new = list(x[1])
                    # brand new
                    if len(old) == 0:
                        return new[0]
                    # no update
                    if len(new) == 0:
                        return old[0]
                    # both old and new: return new
                    return new[0]

                return old_state.cogroup(new_state).mapValues(combiner)

            # group new items and state under the key
            # gives (K, (iterator of state, iterator of event))
            #print 'my-state: {}'.format(old_state.count())
            g = old_state.cogroup(stream_rdd.partitionBy(num_partitions), num_partitions)

            # filter out groups with no events
            # kv[1] is the value, kv[1][1] is the iterator of event
            g = g.filter(lambda kv: len(list(kv[1][1])) > 0)

            # unravel the results of cogroup into values and state
            # the conditional accounts for state that has no existing value for the key
            g = g.mapValues(lambda events_state: (list(events_state[1]), list(events_state[0])[0]
                if len(events_state[0]) else None))

            # apply the user-supplied update function to the events and state
            # this would be mapValues, but we need the key also
            def f1(key_events_state):
                return (key_events_state[0],  # key
                        update_fn(key_events_state[0],  # key
                            key_events_state[1][0],  # events
                            key_events_state[1][1]))    # state
            def f2(partitions):
                # get state from global
                res = map(f1, partitions)
                # save state in global
                return res
            result_state = g.map(f1, preservesPartitioning=True)
#            result_state = g.mapPartitions(f2, preservesPartitioning=True)

            # extract the state and remove items where the state is now None
            new_state = result_state.mapValues(lambda x: x[1]).filter(lambda x: x is not None)
            merged_state = merge_state(old_state, new_state)
            print 'merged_state 1: {}'.format(merged_state.count())
            # extract the transformed events
            res = result_state.flatMapValues(lambda x: x[0])

            checkpoint_counter += 1
            batch_counter += 1
            if self.verbose:
                print 'batch: {}  stream: {}  res {}'.format(batch_counter, stream_rdd.count(), res.count())
            if checkpoint_counter >= checkpoint_interval:
                #if self.verbose:
                print 'saving {} {}'.format(checkpoint_counter, checkpoint_path)
                checkpoint_counter = 0
                XStateImpl.safe_checkpoint(merged_state, checkpoint_path)
                merged_state = CommonSparkContext().spark_context().pickleFile(checkpoint_path)
            #print 'merged_state 2: {}'.format(merged_state.count())
            globals()[instance_name] = merged_state
            globals()['checkpoint_counter'] = checkpoint_counter
            globals()['batch_counter'] = batch_counter
            return res

        # Here is where the augmented update function is applied to the RDDs in the stream.
        # This returns the transformed event dstream, in pair-rdd form
        res = dstream.transform(lambda rdd: transform_func(rdd))

        # Get the value, throw away the key.
        res = res.map(lambda kv: kv[1])
        # Save a copy of the state for next time, where it becomes the initial_state.
        # Return the state also
        return XStreamImpl(res, column_names=column_names, column_types=column_types)
