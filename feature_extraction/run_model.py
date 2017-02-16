import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
import numpy as np

from importlib import reload
from sys import stdin, stdout

from new_model import Seq2SeqModel, train_on_copy_task
from process_data import import_data

def run_model(data):
    tf.reset_default_graph()
    tf.set_random_seed(123)

    with tf.Session() as session:

        # with bidirectional encoder, decoder state size should be
        # 2x encoder state size
        model = Seq2SeqModel(encoder_cell=LSTMCell(20),
                             decoder_cell=LSTMCell(20),
                             max_sequence_length=len(data[0]),
                             seq_width=2,
                             batch_size=100)

        session.run(tf.global_variables_initializer())

        train_on_copy_task(session, model, data,
                           batch_size=100,
                           batches_in_epoch=100,
                           verbose=True)

cache_data, labels = None, None
if __name__ == '__main__':
    stdout.write("To re-run the model, press enter and to exit press CTRL-C\n")
    try:
        while True:
            if cache_data is None:
                cache_data, labels  = import_data()

            stdout.write("Training on data...\n")
            run_model(cache_data)
            stdout.write("Finished running model.")

            # Wait for enter
            stdin.readline()

            # Reload the source code
            reload(new_model)

    except KeyboardInterrupt:
        stdout.write("Interrupted, this might take a while...\n")
        exit(0)
