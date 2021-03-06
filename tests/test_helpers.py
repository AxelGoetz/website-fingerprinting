import numpy as np
import unittest

from feature_generation import helpers

class HelperTests(unittest.TestCase):

    def setUp(self):
        self.data = np.array([[1, 2, 3, 4, 0], [1, 0, 0, 0, 0], [3, 4, 5, 0, 0], [40, 3, 2, 0, 0]])
        self.sequence_lengths = np.array([4, 1, 3, 3])
        self.labels = np.array([-1, 1, 2, 3])

    def testEOS(self):
        helpers.add_EOS(self.data, self.sequence_lengths)

        self.assertTrue((self.data[0] == np.array([1, 2, 3, 4, -1])).all())
        self.assertTrue((self.data[1] == np.array([1, -1, 0, 0, 0])).all())
        self.assertTrue((self.data[2] == np.array([3, 4, 5, -1, 0])).all())
        self.assertTrue((self.data[3] == np.array([40, 3, 2, -1, 0])).all())

    def test_time_major(self):
        new_data = helpers.time_major(self.data)

        self.assertTrue((new_data[0] == np.array([1, 1, 3, 40])).all())
        self.assertTrue((new_data[1] == np.array([2, 0, 4, 3])).all())
        self.assertTrue((new_data[2] == np.array([3, 0, 5, 2])).all())
        self.assertTrue((new_data[3] == np.array([4, 0, 0, 0])).all())
        self.assertTrue((new_data[4] == np.array([0, 0, 0, 0])).all())

    def test_shuffle_data(self):
        data_batch = helpers.get_batches(self.data, batch_size=2)
        sequence_lengths_batch = helpers.get_batches(self.sequence_lengths, batch_size=2)
        labels_batch = helpers.get_batches(self.labels, batch_size=2)

        r_data = next(data_batch)
        r_sequence_length = next(sequence_lengths_batch)
        r_labels = next(labels_batch)

        for i, val in enumerate(r_data):
            if val[0] == 40:
                self.assertEqual(r_sequence_length[i], 3)
                self.assertEqual(r_labels[i], 3)

    def test_get_batches(self):
        self.data = [1] * 101
        iterator = helpers.get_batches(self.data, batch_size=100)

        count = 0
        for i in iterator:
            count += 1

        self.assertEqual(count, 1)

    def test_pad_traces1(self):
        self.data = [[[1.1, -1], [2, 1], [3, 1], [4, 1]], [[1, -1]]]

        padded_matrix, sequence_lengths = helpers.pad_traces(self.data, extra_padding=5)

        self.assertEqual(len(padded_matrix[0]), 9)
        self.assertEqual(sequence_lengths[0], 4)

        self.assertEqual(len(padded_matrix[1]), 9)
        self.assertEqual(sequence_lengths[1], 1)

    def test_pad_traces2(self):
        self.data = [[[1.1, -1], [2, 1], [3, 1], [4, 1]], [[1, -1]]]

        padded_matrix, sequence_lengths = helpers.pad_traces(self.data, extra_padding=5, reverse=True)

        self.assertEqual(len(padded_matrix[0]), 9)
        self.assertEqual(sequence_lengths[0], 4)
        self.assertEqual(padded_matrix[0][0][0], 4)
        self.assertEqual(padded_matrix[0][0][1], 1)


        self.assertEqual(len(padded_matrix[1]), 9)
        self.assertEqual(sequence_lengths[1], 1)
        self.assertEqual(padded_matrix[1][0][0], 1)
        self.assertEqual(padded_matrix[1][0][1], -1)

    def test_extract_filename(self):
        paths = [
            "/Users/test/Desktop/file.txt",
            "Hello/This/should/work/too",
            "file.txt"
        ]

        extensions = ["txt", "", "txt"]

        results = [helpers.extract_filename_from_path(path, extension) for (path, extension) in zip(paths, extensions)]

        self.assertEqual(results[0], "file")
        self.assertEqual(results[1], "too")
        self.assertEqual(results[2], "file")
