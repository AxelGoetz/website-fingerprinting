from os import scandir, makedirs, path as ospath
from sys import stdout, path, exit

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
    Creates a dir if it does not exists, if it exists, it deletes it and creates a new one

    @param directory is a string containing the **absolute** path to the directory
    """
    if ospath.exists(directory):
        from shutil import rmtree
        rmtree(directory)

    makedirs(directory)

def extract_features_from_files(feature_extraction, paths, save_dir, extension=".cell", model_name=""):
    """
    Given a list of paths, extract the features from these paths and stores them in the appropriate directory and file

    @param feature_extraction is a function that takes a trace as an input and returns a list of features (1D)
    @param path is a list of paths to the raw cell files
    @param save_dir is the directory where you save the features for the traces.
        Every in this dir is called `{website}-{id}.cellf` with both `website` and `id` being integers
    @param extension is the extension of the files that contain the raw traces
    @param model_name is used for printing for what model we are extracting features
    """
    if model_name == "":
        stdout.write("Starting to extract features\n")
    else:
        stdout.write("Starting to extract features for {}\n".format(model_name))

    create_dir_if_not_exists(save_dir)
    total_files = len(paths)

    for i, path in enumerate(paths):
        try:
            file_content = read_cell_file(path)
        except:
            print("Cannot read file: " + path)
            exit(0)

        features = feature_extraction(file_content)

        file_name = ospath.basename(path)
        file_name = file_name.replace(extension, "")
        file_name += ".cellf"

        write_to_file(features, save_dir, file_name)

        if i % 50 == 0:
            update_progess(total_files, i)


def extract_features(feature_extraction, save_dir, data_dir=DATA_DIR, extension=".cell", model_name=""):
    """
    For all of the files in the in the `data_dir` with the `extension` extension, it extracts the features using the `feature_extraction` function.

    @param feature_extraction is a function that takes a trace as an input and returns a list of features (1D)
    @param save_dir is the directory where you save the features for the traces.
        Every in this dir is called `{website}-{id}.cellf` with both `website` and `id` being integers
    @param data_dir is the absolute path to the data directory
    @param extension is the extension of the files that contain the raw traces
    @param model_name is used for printing for what model we are extracting features
    """
    paths = []
    for i, f in enumerate(scandir(data_dir)):
        if f.is_file() and f.name[-len(extension):] == extension:
            paths.append(f.path)

    extract_features_from_files(feature_extraction, paths, save_dir, extension=extension, model_name=model_name)


def extract_partial_features(file_path, save_dir, data_dir=DATA_DIR, extension=".cell"):
    """
    Only extracts the features from the paths in the `file_path` file
    """
    from naive_bayes import extract_nb_features
    from random_forest import extract_rf_features
    from svc1 import extract_svc1_features
    from svc2 import extract_svc2_features
    import subprocess

    paths = ''
    with open(file_path, 'r') as f:
        paths = f.readline().split(' ')

    create_dir_if_not_exists(save_dir + '/knn_cells/')

    subprocess.run([
        'go', 'run', dirname + '/kNN.go', '-folder', data_dir + '/',
        '-new_path', save_dir + '/knn_cells/', '-extension', extension, 'all-files']
    )


    extract_features_from_files(extract_nb_features, paths, save_dir + '/nb_cells', extension=extension, model_name="naive bayes")
    extract_features_from_files(extract_rf_features, paths, save_dir + '/rf_cells', extension=extension, model_name="random forest")
    extract_features_from_files(extract_svc1_features, paths, save_dir + '/svc1_cells', extension=extension, model_name="svc1")
    extract_features_from_files(extract_svc2_features, paths, save_dir + '/svc2_cells', extension=extension, model_name="svc2")

    stdout.write("Finished extracting features\n")

def extract_all_features(save_dir, data_dir=DATA_DIR, extension=".cell"):
    from naive_bayes import extract_nb_features
    from random_forest import extract_rf_features
    from svc1 import extract_svc1_features
    from svc2 import extract_svc2_features
    import subprocess

    create_dir_if_not_exists(save_dir + '/knn_cells/')
    subprocess.run([
        'go', 'run', dirname + '/kNN.go', '-folder', data_dir + '/',
        '-new_path', save_dir + '/knn_cells/', '-extension', extension]
    )

    extract_features(extract_nb_features, save_dir + '/nb_cells', data_dir=data_dir, extension=extension, model_name="naive bayes")
    extract_features(extract_rf_features, save_dir + '/rf_cells', data_dir=data_dir, extension=extension, model_name="random forest")
    extract_features(extract_svc1_features, save_dir + '/svc1_cells', data_dir=data_dir, extension=extension, model_name="svc1")
    extract_features(extract_svc2_features, save_dir + '/svc2_cells', data_dir=data_dir, extension=extension, model_name="svc2")

    stdout.write("Finished extracting features\n")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Extracts the features from the cells")

    parser.add_argument('--all_files', action='store_true', help="Whether to process all files or just the ones in `../data/X_test` (default false)")
    parser.add_argument('--extension', metavar='', help="Extension of the cell files", default=".cell")

    args = parser.parse_args()
    if args.all_files:
        extract_all_features(DATA_DIR + '/..', DATA_DIR, extension=args.extension)
    else:
        extract_partial_features(DATA_DIR + '/../X_test', DATA_DIR + '/..', DATA_DIR, extension=args.extension)
