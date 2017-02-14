"""
We make the assumption that the amount of website we need to classify is only limited (only 110 in our dataset).
Therefore there is no need to use a sampled softmax to handle a large amount of output classes as is often done for NLP problems.

This file implements a RNN encoder-decoder model (also known as sequence-to-sequence models).
Then in order to extract the features, we use the output vector from the encoder model.

We made the choice not to implement an attention mechanism (which means that the decoder is allowed to have a 'peak' at the input).
The reason why is because we are not trying to maximize the output of the decoder but instead the feature selection process.
(http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/)

NOTE: The model can be adapted to different batch sizes and sequence lengths without retraining
(e.g. by serializing model parameters and Graph definitions via tf.train.Saver)
but changing vocabulary size requires retraining the model.

We will use time-major rather than batch-major as it is slightly more efficient.

! Backpropagation through time
! GRU vs LSTM
! Train the model on the output of the previous, on the vector as input or on the actual data
! Does encoder share weights with decoder or not (Less computation vs natural (https://arxiv.org/pdf/1409.3215.pdf))
! Reverse traces? (https://arxiv.org/pdf/1409.3215.pdf)
! Deep LSTM or GRU networks for each cell?

Hyperparameters to tune:
------------------------
-


Thanks to https://github.com/ematvey/tensorflow-seq2seq-tutorials
"""


import numpy as np
import tensorflow as tf
import helpers

from tensorflow.contrib.rnn import (LSTMCell, LSTMStateTuple,
                                    InputProjectionWrapper,
                                    OutputProjectionWrapper)

class Seq2SeqModel():
    """
    # TODO: Fill in
    """

    PAD = 0
    EOS = 1

    def __init__(self, encoder_cell, decoder_cell, max_sequence_length=100, input_embedding_size=20, batch_size=100):
        """
        @param encoder_cell is a `rnn_cell` used in the encoder
        @param decoder_cell is a `rnn_cell` used in the decoder
        @param max_sequence_length is the max sequence length possible
        @param input_embedding_size
        @param batch_size
        """
        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self.max_sequence_length = max_sequence_length
        self.input_embedding_size = input_embedding_size
        self.batch_size = batch_size

        self._make_graph()

    def _make_graph(self):
        self._init_placeholders()
        self._init_encoder_projection()
        self._init_encoder()

        self.eos_time_slice = tf.one_hot(
            tf.ones([self.batch_size],
                    dtype=tf.int32, name='EOS'),
            2, name='EOS_OneHot')

        self.pad_time_slice = tf.one_hot(
            tf.zeros([self.batch_size],
                     dtype=tf.int32, name='PAD'),
            2, name='PAD_OneHot')

        self._init_decoder()

        self._init_train()

    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(shape=(self.max_sequence_length, self.batch_size, 2), dtype=tf.float32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32, name='encoder_inputs_length')

        # Shaped max_sequence_length because padding should already be added
        self.decoder_targets = tf.placeholder(shape=(self.max_sequence_length, self.batch_size, 2), dtype=tf.float32, name='decoder_targets')

    def _init_encoder_projection(self):
        with tf.variable_scope('EncoderInputProjection') as scope:
            self.encoder_inputs_projected = self.projection(self.encoder_inputs, self.input_embedding_size, scope)

    def _init_encoder(self):
        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
            self.encoder_cell, self.encoder_inputs_projected,
            dtype=tf.float32, time_major=True,
        )

    def _init_decoder(self):
        # Batch dimensions are dynamic, i.e. they can change in runtime, from batch to batch
        encoder_max_time, batch_size, _ = tf.unstack(tf.shape(self.encoder_inputs))

        # how far to run the decoder is our decision
        self.decoder_lengths = self.encoder_inputs_length

        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self.decoder_cell, self.loop_fn)
        self.decoder_outputs = decoder_outputs_ta.stack()

        with tf.variable_scope('DecoderOutputProjection') as scope:
            self.decoder_outputs = self.projection(self.decoder_outputs, 2, scope)

    def _init_train(self):
        self.loss = tf.reduce_sum(tf.square(self.decoder_targets - self.decoder_outputs))
        self.train_op = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

    def projection(self, inputs, projection_size, scope):
        """
        Args:
            inputs: shape like [time, batch, input_size] or [batch, input_size]
            projection_size: int32
            scope: outer variable scope
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

    def loop_fn_initial(self, time, cell_output, cell_state, loop_state):
        assert cell_output is None and loop_state is None and cell_state is None

        elements_finished = (time >= self.decoder_lengths)  # all True at the 1st step
        with tf.variable_scope('DecoderInputProjection') as scope:
            initial_input = self.projection(self.eos_time_slice, self.input_embedding_size, scope)
        initial_cell_state = self.encoder_final_state
        initial_loop_state = None  # we don't need to pass any additional information

        return (elements_finished,
                initial_input,
                initial_cell_state,
                None,  # cell output is dummy here
                initial_loop_state)

    def loop_fn(self, time, cell_output, cell_state, loop_state):
        """ loop_fn determines transitions between RNN unroll steps
        """

        if cell_state is None:    # time == 0
            return self.loop_fn_initial(time, cell_output, cell_state, loop_state)

        emit_output = cell_output  # == None for time == 0

        next_cell_state = cell_state

        elements_finished = (time >= self.decoder_lengths)
        finished = tf.reduce_all(elements_finished)

        def padded_next_input():
            with tf.variable_scope('DecoderInputProjection', reuse=True) as scope:
                return self.projection(self.pad_time_slice, self.input_embedding_size, scope)

        def search_for_next_input():
            """ output->input transition:

                output[t] -> output projection[t] -> prediction[t] ->
                -> input[t+1] -> input projection[t+1]
            """
            with tf.variable_scope('DecoderOutputProjection') as scope:
                output = self.projection(cell_output, 2, scope)
            prediction = output
            with tf.variable_scope('DecoderInputProjection', reuse=True) as scope:
                projection_ = self.projection(prediction, self.input_embedding_size, scope)
            return projection_

        next_input = tf.cond(finished, padded_next_input, search_for_next_input)

        next_loop_state = None

        result = (elements_finished,
                next_input,
                next_cell_state,
                emit_output,
                next_loop_state)

        return result

    def next_batch(self, batches, sequence_lengths):
        """
        Returns the next batch.

        @batches an iterator with all of the batches (in batch-major form with padding)
        @sequence_lengths an iterator with the sequence lengths
        """
        batch = next(batches)
        encoder_input_lengths_ = next(sequence_lengths)

        encoder_inputs_ = helpers.time_major(batch)
        decoder_targets_ = helpers.time_major(helpers.add_EOS(batch, encoder_input_lengths_))

        print(encoder_inputs_.shape)
        print(decoder_targets_.shape)
        print(encoder_input_lengths_)

        return {
            self.encoder_inputs: encoder_inputs_,
            self.encoder_inputs_length: encoder_input_lengths_,
            self.decoder_targets: decoder_targets_,
        }

def train_on_copy_task(sess, model, data, sequence_lengths,
                       batch_size=100,
                       max_batches=5000,
                       batches_in_epoch=1000,
                       verbose=True):
    """
    TODO
    """

    batches = helpers.get_batches(data, batch_size=batch_size)
    sequence_lengths_batch = helpers.get_batches(sequence_lengths, batch_size=batch_size)

    loss_track = []

    try:
        for batch in range(max_batches):
            fd = model.next_batch(batches, sequence_lengths_batch)
            _, l = sess.run([model.train_op, model.loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                print('  minibatch loss: {}'.format(sess.run(model.loss, fd)))
                predict_ = sess.run(model.decoder_outputs, fd)
                for i, (inp, pred) in enumerate(zip(fd[model.encoder_inputs].T, predict_.T)):
                    print('  sample {}:'.format(i + 1))
                    print('    input     > {}'.format(inp))
                    print('    predicted > {}'.format(pred))
                    if i >= 2:
                        break
                print()

    except KeyboardInterrupt:
        print('training interrupted')

    return loss_track
