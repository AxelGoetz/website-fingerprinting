"""
This file implements a RNN encoder-decoder model (also known as sequence-to-sequence models).

We made the choice not to implement an attention mechanism (which means that the decoder is allowed to have a 'peak' at the input).
The reason why is because we are not trying to maximize the output of the decoder but instead the feature selection process.
(http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/)

We will use batch-major rather than time-major even though time-major is slightly more efficient
since it makes the feature extraction process a lot easier.

We will not be using bucketing because traces of the same webpage will have the same length.
Therefore every batch, we will most likely be training the seq2seq model on one webpage

! Does encoder share weights with decoder or not (Less computation vs natural (https://arxiv.org/pdf/1409.3215.pdf))
! Reverse traces (https://arxiv.org/pdf/1409.3215.pdf)

Hyperparameters to tune:
------------------------
- Learning rate
- Which cell to use (GRU vs LSTM) or a deep RNN architecture using `MultiRNNCell`
- Reversing traces
- Bidirectional encoder
- Other objective functions (such as MSE,...)
- Amount of encoder and decoder hidden states
"""
import numpy as np
import tensorflow as tf

from sys import stdout, path
from os import path as ospath

from tensorflow.contrib.rnn import LSTMStateTuple

path.append(ospath.dirname(ospath.dirname(ospath.abspath(__file__))))
import helpers


class Seq2SeqModel():
    """
    Implements a sequence to sequence model for real values

    Attributes:
        - encoder_cell is the cell that will be used for encoding
            (Should be part of `tf.nn.rnn_cell`)
        - decoder cell is the cell used for decoding
            (Should be part of `tf.nn.rnn_cell`)

        - seq_width shows how many features each input in the sequence has
            (For website fingerprinting this is only 2 (packet_size, incoming))
        - batch_size

        - bidirectional is a boolean value that determines whether the encoder is bidirectional or not
        - reverse is also a boolean value that when if true, reversed the traces for training
    """

    def __init__(self, encoder_cell, decoder_cell, seq_width, batch_size=100, bidirectional=False, reverse=False, saved_graph=None, sess=None, learning_rate=0.0006):
        """
        @param saved_graph is a string, representing the path to the saved graph
        """
        # Constants
        self.PAD = 0
        self.EOS = -1

        self.reverse = reverse
        self.seq_width = seq_width
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.bidirectional = bidirectional

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self._make_graph()

        if saved_graph is not None and sess is not None:
            self.import_from_file(sess, saved_graph)

    def _make_graph(self):
        """
        Construct the graph
        """

        self._init_placeholders()

        self._init_encoder()
        self._init_decoder()

        self._init_train()

    def _init_placeholders(self):
        """
        The main placeholders used for the input data, and output
        """
        # The usual format is: `[self.batch_size, max_sequence_length, self.seq_width]`
        # But we define `max_sequence_length` as None to make it dynamic so we only need to pad
        # each batch to the maximum sequence length
        self.encoder_inputs = tf.placeholder(tf.float32,
            [self.batch_size, None, self.seq_width])

        self.encoder_inputs_length = tf.placeholder(tf.int32, [self.batch_size])

        self.decoder_targets = tf.placeholder(tf.float32,
            [self.batch_size, None, self.seq_width])

    def _init_encoder(self):
        """
        Creates the encoder attributes

        Attributes:
            - encoder_outputs is shaped [max_sequence_length, batch_size, seq_width]
                (since time-major == True)
            - encoder_final_state is shaped [batch_size, encoder_cell.state_size]
        """
        if not self.bidirectional:
            with tf.variable_scope('Encoder') as scope:
                self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                    cell=self.encoder_cell,
                    dtype=tf.float32,
                    sequence_length=self.encoder_inputs_length,
                    inputs=self.encoder_inputs,
                    time_major=False)
        else:
            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_final_state,
              encoder_bw_final_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs,
                                                sequence_length=self.encoder_inputs_length,
                                                dtype=tf.float32, time_major=False)
                )

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_fw_outputs), 2)

            encoder_final_state_c = tf.concat(
                (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

            encoder_final_state_h = tf.concat(
                (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

            self.encoder_final_state = LSTMStateTuple(
                c=encoder_final_state_c,
                h=encoder_final_state_h
            )

    def _init_decoder(self):
        """
        Creates decoder attributes.
        We cannot simply use a dynamic_rnn since we are feeding the outputs of the
        decoder back into the inputs.
        Therefore we use a raw_rnn and emulate a dynamic_rnn with this behavior.
        (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py)
        """
        # EOS token added
        self.decoder_inputs_length = self.encoder_inputs_length + 1

        def loop_fn_initial(time, cell_output, cell_state, loop_state):
            elements_finished = (time >= self.decoder_inputs_length)

            # EOS token (0 + self.EOS)
            initial_input = tf.zeros([self.batch_size, self.decoder_cell.output_size], dtype=tf.float32) + self.EOS
            initial_cell_state = self.encoder_final_state
            initial_loop_state = None  # we don't need to pass any additional information

            return (elements_finished,
                    initial_input,
                    initial_cell_state,
                    None,  # cell output is dummy here
                    initial_loop_state)

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:  # time == 0
                return loop_fn_initial(time, cell_output, cell_state, loop_state)

            cell_output.set_shape([self.batch_size, self.decoder_cell.output_size])

            emit_output = cell_output

            next_cell_state = cell_state

            elements_finished = (time >= self.decoder_inputs_length)
            finished = tf.reduce_all(elements_finished)

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([self.batch_size, self.decoder_cell.output_size], dtype=tf.float32), # self.PAD
                lambda: cell_output # Use the input from the previous cell
            )

            next_loop_state = None

            return (
                elements_finished,
                next_input,
                next_cell_state,
                emit_output,
                next_loop_state
            )

        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self.decoder_cell, loop_fn)
        self.decoder_outputs = decoder_outputs_ta.stack()
        self.decoder_outputs = tf.transpose(self.decoder_outputs, [1, 0, 2])

        with tf.variable_scope('DecoderOutputProjection') as scope:
            self.decoder_outputs = self.projection(self.decoder_outputs, self.seq_width, scope)

    def _init_train(self):
        self.loss = tf.reduce_sum(tf.square(self.decoder_targets - self.decoder_outputs))

        # Which optimizer to use? `GradientDescentOptimizer`, `AdamOptimizer` or `RMSProp`?
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def projection(self, inputs, projection_size, scope):
        """
        Projects the input with a known amount of features to a `projection_size amount of features`

        @param inputs is shaped like [time, batch, input_size] or [batch, input_size]
        @param projection_size int32
        @param scope outer variable scope
        """
        input_size = inputs.get_shape()[-1].value

        with tf.variable_scope(scope) as scope:
            W = tf.get_variable(name='W', shape=[input_size, projection_size],
                                dtype=tf.float32)

            b = tf.get_variable(name='b', shape=[projection_size],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0, dtype=tf.float32))

        input_shape = tf.unstack(tf.shape(inputs))

        if len(input_shape) == 3:
            time, batch, _ = input_shape  # dynamic parts of shape
            inputs = tf.reshape(inputs, [-1, input_size])

        elif len(input_shape) == 2:
            batch, _depth = input_shape

        else:
            raise ValueError("Weird input shape: {}".format(inputs))

        linear = tf.add(tf.matmul(inputs, W), b)

        if len(input_shape) == 3:
            linear = tf.reshape(linear, [time, batch, projection_size])

        return linear

    def next_batch(self, batches, in_memory, max_time_diff=float("inf")):
        """
        Returns the next batch.

        @param batches an iterator with all of the batches (
            if in_memory == True:
                in batch-major form without padding
            else:
                A list of paths to the files
        )
        @param in_memory is a boolean value
        @param max_time_diff **(should only be defined if `in_memory == False`)**
            specifies what the maximum time different between the first packet in the trace and the last one should be

        @return if in_memory is False, returns a tuple of (dict, [paths], max_length) where paths is a list of paths for each batch
            else it returns a dict for training
        """
        batch = next(batches)
        data_batch = batch

        if not in_memory:
            data_batch = [helpers.read_cell_file(path, max_time_diff=max_time_diff) for path in batch]
            for i, cell in enumerate(data_batch):
                data_batch[i] = [packet[0] * packet[1] for packet in cell]

        data_batch, encoder_input_lengths_ = helpers.pad_traces(data_batch, reverse=self.reverse, seq_width=self.seq_width)
        encoder_inputs_ = data_batch

        decoder_targets_ = helpers.add_EOS(data_batch, encoder_input_lengths_)

        train_dict = {
            self.encoder_inputs: encoder_inputs_,
            self.encoder_inputs_length: encoder_input_lengths_,
            self.decoder_targets: decoder_targets_,
        }

        if not in_memory:
            return (train_dict, batch, max(encoder_input_lengths_))
        return train_dict

    def save(self, sess, file_name):
        """
        Save the model in a file

        @param sess is the session
        @param file_name is the file name without the extension
        """
        saver = tf.train.Saver()
        saver.save(sess, file_name)
        # saver.export_meta_graph(filename=file_name + '.meta')

    def import_from_file(self, sess, file_name):
        """
        Imports the graph from a file

        @param sess is the session
        @param file_name is a string that represents the file name
            without the extension
        """

        # Get the graph
        saver = tf.train.Saver()

        # Restore the variables
        saver.restore(sess, file_name)


def train_on_copy_task(sess, model, data,
                       batch_size=100,
                       max_batches=None,
                       batches_in_epoch=1000,
                       max_time_diff=float("inf"),
                       verbose=False):
    """
    Train the `Seq2SeqModel` on a copy task

    @param sess is a tensorflow session
    @param model is the seq2seq model
    @param data is the data (in batch-major form and not padded or a list of files (depending on `in_memory`))
    """
    batches = helpers.get_batches(data, batch_size=batch_size)

    loss_track = []

    batches_in_data = len(data) // batch_size
    if max_batches is None or batches_in_data < max_batches:
        max_batches = batches_in_data - 1

    try:
        for batch in range(max_batches):
            print("Batch {}/{}".format(batch, max_batches))
            fd, _, length = model.next_batch(batches, False, max_time_diff)
            _, l = sess.run([model.train_op, model.loss], fd)
            loss_track.append(l / length)

            if batch == 0 or batch % batches_in_epoch == 0:
                model.save(sess, 'seq2seq_model')
                helpers.save_object(loss_track, 'loss_track.pkl')

                if verbose:
                    stdout.write('  minibatch loss: {}\n'.format(sess.run(model.loss, fd)))
                    predict_ = sess.run(model.decoder_outputs, fd)
                    for i, (inp, pred) in enumerate(zip(fd[model.encoder_inputs].swapaxes(0, 1), predict_.swapaxes(0, 1))):
                        stdout.write('  sample {}:\n'.format(i + 1))
                        stdout.write('    input     > {}\n'.format(inp))
                        stdout.write('    predicted > {}\n'.format(pred))
                        if i >= 0:
                            break
                    stdout.write('\n')

    except KeyboardInterrupt:
        stdout.write('training interrupted')
        model.save(sess, 'seq2seq_model')
        exit(0)

    model.save(sess, 'seq2seq_model')
    helpers.save_object(loss_track, 'loss_track.pkl')

    return loss_track

def get_vector_representations(sess, model, data, save_dir,
                       batch_size=100,
                       max_batches=None,
                       batches_in_epoch=1000,
                       max_time_diff=float("inf"),
                       extension=".cell"):
    """
    Given a trained model, gets a vector representation for the traces in batch

    @param sess is a tensorflow session
    @param model is the seq2seq model
    @param data is the data (in batch-major form and not padded or a list of files (depending on `in_memory`))
    """
    batches = helpers.get_batches(data, batch_size=batch_size)

    batches_in_data = len(data) // batch_size
    if max_batches is None or batches_in_data < max_batches:
        max_batches = batches_in_data - 1

    try:
        for batch in range(max_batches):
            print("Batch {}/{}".format(batch, max_batches))
            fd, paths, _ = model.next_batch(batches, False, max_time_diff)
            l = sess.run(model.encoder_final_state, fd)

            # Returns a tuple, so we concatenate
            l = np.concatenate((l.c, l.h), axis=1)

            file_names = [helpers.extract_filename_from_path(path, extension) for path in paths]

            for file_name, features in zip(file_names, list(l)):
                helpers.write_to_file(features, save_dir, file_name, new_extension=".cellf")

    except KeyboardInterrupt:
        stdout.write('Interrupted')
        exit(0)

    return results
