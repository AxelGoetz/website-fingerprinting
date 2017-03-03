from os import scandir, makedirs, path as ospath
from sys import stdout, path

# Hack to import from sibling directory
path.append(ospath.dirname(path[0]))

from feature_generation.helpers import read_cell_file

dirname, _ = ospath.split(ospath.abspath(__file__))
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
    content = list(map(lambda x: str(x), content))

    with open(save_dir + '/' + file_name, 'w') as f:
        f.write(' '.join(content))

def create_dir_if_not_exists(directory):
    """
    Creates a dir if it does not exists

    @param directory is a string containing the **absolute** path to the directory
    """
    if not ospath.exists(directory):
        makedirs(directory)

def extract_features(feature_extraction, save_dir, data_dir=DATA_DIR, extension=".cell", model_name=""):
    """
    For all of the files in the in the `data_dir` with the `extension` extension, it extracts the features using the `feature_extraction` function.

    @param feature_extraction is a function that takes a trace as an input and returns a list of features (1D)
    @param save_dir is the directory where you save the features for the traces.
        Every in this dir is called `{website}-{id}.cellf` with both `website` and `id` being integers
    @param data_dir is the absolute path to the data directory
    @param extension is the extension of the files that contain the raw traces
    @param feature_name is used for printing for what model we are extracting features
    """
    if model_name == "":
        stdout.write("Starting to extract features\n")
    else:
        stdout.write("Starting to extract features for {}\n".format(model_name))

    create_dir_if_not_exists(save_dir)

    total_files = len([f for f in scandir(data_dir) if f.is_file()])

    for i, f in enumerate(scandir(data_dir)):
        if f.is_file() and f.name[-len(extension):] == extension:
            file_content = read_cell_file(f.path)
            features = feature_extraction(file_content)

            file_name = f.name
            file_name = file_name.replace(extension, "")
            file_name += ".cellf"

            write_to_file(features, save_dir, file_name)

        if i % 50 == 0:
            update_progess(total_files, i)


def extract_all_features(save_dir, data_dir=DATA_DIR):
    from kNN import extract_kNN_features
    from naive_bayes import extract_nb_features
    from random_forest import extract_rf_features
    from svc1 import extract_svc1_features
    from svc2 import extract_svc2_features

    extract_features(extract_kNN_features, save_dir + '/knn_cells', data_dir=data_dir, extension=".cell", model_name="kNN")
    extract_features(extract_nb_features, save_dir + '/nb_cells', data_dir=data_dir, extension=".cell", model_name="naive bayes")
    extract_features(extract_rf_features, save_dir + '/rf_cells', data_dir=data_dir, extension=".cell", model_name="random forest")
    extract_features(extract_svc1_features, save_dir + '/svc1_cells', data_dir=data_dir, extension=".cell", model_name="svc1")
    extract_features(extract_svc2_features, save_dir + '/svc2_cells', data_dir=data_dir, extension=".cell", model_name="svc2")

    stdout.write("Finished extracting features\n")

if __name__ == '__main__':
    extract_all_features(DATA_DIR + '/..', DATA_DIR)
