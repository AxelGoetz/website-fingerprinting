from os import scandir
from sys import stdout

import constants

def update_progess(total, current):
    """Prints a percentage of how far the process currently is"""
    stdout.write("{:.2f} %\r".format((current/total) * 100))
    stdout.flush()

def read_feature_file(path):
    """
    For a file, reads its contents and returns them in the appropriate format

    @param path is a path to the file

    @return a list of features
    """
    features = []
    with open(path, 'r') as open_file:
        for line in open_file:
            split = line.split()
            split = [float(x) for x in split]
            features.extend(split)

    return features

def pull_data_in_memory(data_dir, extension=".cell"):
    """
    Gets the content of all the data in the `data_dir` directory in memory.

    @return is a tuple with the following format: [(features, webpage_label)]
    """
    data = []
    labels = []

    stdout.write("Starting data import\n")
    total_files = len([f for f in scandir(data_dir) if f.is_file()])

    for i, f in enumerate(scandir(data_dir)):
        if f.is_file() and f.name[-len(extension):] == extension:
            name = f.name
            name = name.replace(extension, "")

            name_split = name.split('-') # Contains [webpage, index (OPTIONAL)]

            webpage_label = None
            # If the length is 1, classify as unknown webpage
            if len(name_split) == 1:
                webpage_label = constants.UNMONITORED_LABEL
            else:
                webpage_label = int(name_split[0])

            labels.append(webpage_label)

            data.append(read_feature_file(f.path))

            if i % 100 == 0:
                update_progess(total_files, i)

    stdout.write("Finished importing data\n")
    return list(zip(data, labels))
