from os import scandir, path
from sys import stdout

from feature_extraction.helpers import read_cell_file

dirname, _ = path.split(path.abspath(__file__))
DATA_DIR = dirname + '/../data/cells'

def update_progess(total, current):
    """Prints a percentage of how far the process currently is"""
    stdout.write("{:.2f} %\r".format((current/total) * 100))
    stdout.flush()

def write_to_file(content, save_dir, file_name):
    """
    Writes the features to a file in a space separated fashion

    @param content is a 1D list of features
    @param save_dir is the absolute path to the directory where we save the traces
    @param file_name is the file name to store the features in
    """
    with open(save_dir + '/' + file_name, 'w') as f:
        f.write(' '.join(content))

def extract_features(feature_extraction, save_dir, data_dir=DATA_DIR, extension=".cell"):
    """
    For all of the files in the in the `data_dir` with the `extension` extension, it extracts the features using the `feature_extraction` function.

    @param feature_extraction is a function that takes a trace as an input and returns a list of features (1D)
    @param save_dir is the directory where you save the features for the traces.
        Every in this dir is called `{website}-{id}.cellf` with both `website` and `id` being integers
    @param data_dir is the absolute path to the data directory
    @param extension is the extension of the files that contain the raw traces
    """
    stdout.write("Starting to extract features")

    total_files = len([f for f in scandir(data_dir) if f.is_file()])

    for i, f in enumerate(scandir(data_dir)):
        if f.is_file() and f.name[-len(extension):] == extension:
            file_content = read_cell_file(f.path)
            features = feature_extraction(file_content)

            file_name = f.name
            file_name = file_name.replace(".cell", "")
            file_name += ".fcell"

            write_to_file(features, save_dir, file_name)

        if i % 100 == 0:
            update_progess(total_files, i)

    stdout.write("Finished extracting features\n")
    return files

if __name__ == '__main__':
    pass
