"""
Includes all of the helper functions such as:
- Getting new batches of data
- Converting data to time-major format
"""

import numpy as np

def time_major(data, max_sequence_length=None):
    """
    Return the data in time-major form (Most often used to turn a single batch into this form)

    @param data is an array of sequences
    @param max_sequence_length is an integer that specifies how large the `max_time` dimension is.
        If the value is none, the maximum sequence length is used

    @return (batch, sequence_length) is a tuple that contains the batch in **time-major** form [max_time, batch_size] padded with the PAD symbol
        and a list of integers that specified the length of each sequence
    """

    sequence_lengths = [len(seq) for seq in data]
    batch_size = len(data)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    # Define the matrix with all zeros (PAD symbol)
    data_time_major = np.zeros(shape=[batch_size, max_sequence_length, 2], dtype=np.float64)

    for i, seq in enumerate(data):
        for j, element in enumerate(seq):
            data_time_major[i, j] = element

    # Swap axis
    data_time_major = data_time_major.swapaxes(0, 1)

    return (data_time_major, sequence_lengths)


def batches(data, batch_size=100):
    """
    Divides the data up into batches

    @param data is an array of sequences
    @param batch_size is the size of each batch

    @return an iterator with the batch sizes
    """
    # TODO: Perhaps we can take the class distribution in each class into consideration

    # Randomly shuffle the data
    np.random.shuffle(data)

    data_length = len(data)

    for i in range(0, data_length, batch_size):
        start_index = i
        end_index = i + batch_size if i + batch_size < data_length else data_length - 1

        yield data[start_index: end_index]
