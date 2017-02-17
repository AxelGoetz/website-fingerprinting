"""
Includes all of the helper functions such as:
- Getting new batches of data
- Converting data to time-major format
"""

import numpy as np
TRACE_DELIMITER = '\t'

def add_EOS(data, sequence_lengths, EOS=-1):
    """
    For the decoder targets, we want to add a EOS symbol after the end of each sequence

    @param data in **batch-major** format
    @param sequence_lengths is a array-like object with the sequence lengths of the elements in the data object
    @param EOS is the end-of-sequence marker
    """
    for i, sequence in enumerate(data):
        sequence[sequence_lengths[i]] = EOS

    return data


def time_major(data):
    """
    Return the data in time-major form (Most often used to turn a single batch into this form)
    *(Assumes the traces are already padded)*

    @param data is an array of sequences
    @return batch contains the batch in **time-major** form [max_time, batch_size] padded with the PAD symbol
    """
    # Swap axis
    return data.swapaxes(0, 1)

def shuffle_data(data, seed=123):
    """
    Shuffles an array-like object, we perform it in here to reset the seed
    and get consistent shuffles.
    """
    np.random.seed(seed)
    np.random.shuffle(data)


def get_batches(data, batch_size=100):
    """
    Divides the data up into batches

    @param data is an array of sequences or an array of paths to files
    @param batch_size is the size of each batch

    @return an iterator with the batch sizes
    """
    # TODO: Perhaps we can take the class distribution in each class into consideration

    # Randomly shuffle the data
    shuffle_data(data)

    data_length = len(data)

    for i in range(0, data_length, batch_size):
        if i + batch_size >= data_length:
            return

        start_index = i
        end_index = i + batch_size

        yield data[start_index: end_index]


def pad_traces(data, extra_padding=1):
    """
    Pad the traces such that they have the same length

    @param data is an 2D matrix in the following format: [[size, incoming]]
    @param extra_padding determines how much extra padding is added to the elements
    @return a tuple where the first element is the padded matrix and the second the lengths
    """
    sequence_lengths = [len(seq) for seq in data]
    batch_size = len(data)

    max_sequence_length = max(sequence_lengths) + extra_padding

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length, 2], dtype=np.float32)

    for i, seq in enumerate(data):
        for j, element in enumerate(seq):
            inputs_batch_major[i][j][0] = element[0]
            inputs_batch_major[i][j][1] = element[1]

    return inputs_batch_major, sequence_lengths

def read_cell_file(path):
    """
    For a file, reads its contents and returns them in the appropriate format

    @param path is a path to the file
    @return a list of (size, incoming pairs)
    """
    contents = []
    with open(path, 'r') as open_file:
        for line in open_file:
            line = line[:-1] # Get rid of newline

            split = line.split(TRACE_DELIMITER)
            split[0] = float(split[0])
            split[1] = int(split[1])

            contents.append(split)

    return contents
