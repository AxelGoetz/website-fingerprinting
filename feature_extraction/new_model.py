"""
This file implements a RNN encoder-decoder model (also known as sequence-to-sequence models).

We made the choice not to implement an attention mechanism (which means that the decoder is allowed to have a 'peak' at the input).
The reason why is because we are not trying to maximize the output of the decoder but instead the feature selection process.
(http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/)

We make the assumption that the amount of website we need to classify is only limited (only 110 in our dataset).
Therefore there is no need to use a sampled softmax to handle a large amount of output classes as is often done for NLP problems.

We will use time-major rather than batch-major as it is slightly more efficient.

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
import helpers

from sys import stdout

from tensorflow.contrib.rnn import LSTMStateTuple

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

    def __init__(self, encoder_cell, decoder_cell, seq_width, batch_size=100, bidirectional=False, reverse=False):
        # Constants
        self.EOS = -1
        self.PAD = 0

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self.seq_width = seq_width

        self.batch_size = batch_size

        self.bidirectional = bidirectional
        self.reverse = reverse

        self._make_graph()

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
        # The usual format is: `[max_sequence_length, self.batch_size, self.seq_width]`
        # But we define `max_sequence_length` as None to make it dynamic so we only need to pad
        # each batch to the maximum sequence length
        self.encoder_inputs = tf.placeholder(tf.float32,
            [None, self.batch_size, self.seq_width])

        self.encoder_inputs_length = tf.placeholder(tf.int32, [self.batch_size])

        self.decoder_targets = tf.placeholder(tf.float32,
            [None, self.batch_size, self.seq_width])

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
                    time_major=True)
        else:
            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_final_state,
              encoder_bw_final_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs,
                                                sequence_length=self.encoder_inputs_length,
                                                dtype=tf.float32, time_major=True)
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

        with tf.variable_scope('DecoderOutputProjection') as scope:
            self.decoder_outputs = self.projection(self.decoder_outputs, self.seq_width, scope)

    def _init_train(self):
        self.loss = tf.reduce_sum(tf.square(self.decoder_targets - self.decoder_outputs))

        # TODO: Which optimizer to use? `GradientDescentOptimizer`, `AdamOptimizer` or `RMSProp`?
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

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

    def next_batch(self, batches, in_memory):
        """
        Returns the next batch.

        @param batches an iterator with all of the batches (
            if in_memory == True:
                in batch-major form without padding
            else:
                A list of paths to the files
        )
        @param in_memory is a boolean value
        """
        batch = next(batches)

        if not in_memory:
            batch = [helpers.read_cell_file(path) for path in batch]

        batch, encoder_input_lengths_ = helpers.pad_traces(batch, reverse=self.reverse)
        encoder_inputs_ = helpers.time_major(batch)

        decoder_targets_ = helpers.time_major(helpers.add_EOS(batch, encoder_input_lengths_))

        return {
            self.encoder_inputs: encoder_inputs_,
            self.encoder_inputs_length: encoder_input_lengths_,
            self.decoder_targets: decoder_targets_,
        }

    def save(self, sess, file_name):
        """
        Save the model in a file

        @param sess is the session
        @param file_name is the file name without the extension
        """
        saver = tf.train.Saver()
        saver.save(sess, file_name + '.ckpt')
        saver.export_meta_graph(filename=file_name + '.meta')

    def import_from_file(self, sess, file_name):
        """
        Imports the graph from a file

        @param sess is the session
        @param file_name is a string that represents the file name
            without the extension
        """

        # Get the graph
        saver = tf.train.import_meta_graph(file_name + '.meta')

        # Restore the variables
        saver.restore(sess, file_name + 'ckpt')


def train_on_copy_task(sess, model, data,
                       batch_size=100,
                       max_batches=None,
                       batches_in_epoch=1000,
                       verbose=True,
                       in_memory=True):
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
        max_batches = batches_in_data

    try:
        for batch in range(max_batches):
            fd = model.next_batch(batches, in_memory)
            _, l = sess.run([model.train_op, model.loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                model.save(sess, 'seq2seq_model')
                if verbose:
                    stdout.write('batch {}\n'.format(batch))
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
