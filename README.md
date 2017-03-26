# Website Fingerprinting [![Build Status](https://travis-ci.com/AxelGoetz/website-fingerprinting.svg?token=MDrK2H5qtb5x5ygwhAzr&branch=master)](https://travis-ci.com/AxelGoetz/website-fingerprinting) [![versioneye](https://www.versioneye.com/user/projects/58b4886a9fd69a003e8d2baa/badge.svg?)](https://www.versioneye.com/user/projects/58b4886a9fd69a003e8d2baa?child=summary#tab-dependencies) [![codecov](https://codecov.io/gh/AxelGoetz/website-fingerprinting/branch/master/graph/badge.svg?token=VhWKyahGjG)](https://codecov.io/gh/AxelGoetz/website-fingerprinting)

There have been a large variety of website fingerprinting attacks.
However most of them require a very tedious process of feature-selection.

There have been some attempts to use autoencoders to solve this problem however those NN require a fixed-length input to begin with.
Hence, you need to perform a feature selection process to begin with.

This project therefore examines the use of RNN's to perform a feature selection process since they can be unrolled to a custom length for each trace.

### Sequence-to-Sequence Model

Essentially, this project implements a sequence-to-sequence model, which has previously mainly been used for natural language processing (NLP) and to perform translation tasks.

![Sequence-to-sequence model image](./static/images/seq2seq.png)

The model consists of two different RNNs, a encoder and a decoder.
First the encoder runs on the input and returns a **thought vector**, which can be thought of as our features.
Next, the decoder uses the thought vector as its initial state and uses it to construct a new sentence as can be seen in the image above.

So if we train a model on a copy task, where it tries to reconstruct the original trace from the thought vector, it has learned to construct a fixed-length representation of a trace, that contains all of the necessary information to represent it.

Therefore the thought vector can be used as features for other machine learning solutions.

### Testing the Outcome

To test if our feature selection process has been effective, we need to compare it with the accuracy of existing hand-picked features.
We will do this by training *existing* models used for website fingerprinting attacks using both the hand-picked features and the automatically generated ones.

Then we will compare both using a wide variety of metrics.

Given time constraints, we will only test our automatically generated features on a small set of (influential) existing models:
- k-fingerprinting attack *(Random Forest)* [1]
- Website Fingerprinting in Onion Routing Based Anonymization Networks *(SVC)* [2]
- Website Fingerprinting at Internet Scale *(SVM with RBF kernel)* [3]
- A Novel Website Fingerprinting Attack against Multi-tab Browsing Behavior *(Naive Bayes)* [4]
- Effective Attacks and Provable Defenses for Website Fingerprinting *(kNN)* [5]

More information on the hand-picked features can be found [here](./feature_extraction/features.md).

Also, some unit tests have been written to test some of the data preprocessing.
All of those can be run by using:

```
python -m unittest discover
```

Next, to generate a new coverage report, we need to install [coverage](https://coverage.readthedocs.io/en/coverage-4.3.4/) and run:
```
pip install coverage # Outside of your virtual environment
coverage run --omit="/usr/local/*" -m unittest discover # Inside the virtual environment
```

## Running the Code
Since some of the source files contain unicode characters, you need to run all of the code with `python3`.

The seq2seq model can be run by using:
```
python feature_generation/run_model.py
```

To extract all of the hand-picked features from the data, first update the relative path in the [feature_extraction.py](./feature_extraction/feature_extraction.py) file to the data.

Next, run:
```
python feature_extraction/feature_extraction.py
```

This script will create a new directory for every model within your data directory with the features inside of `{webpage_index}-{sample_number}.cellf` files.

Finally, to run all of the models, you can run the script:
```
python run_models/run_model.py
```

with the appropriate parameters.

// TODO: Add system manual

## Installation

The seq2seq model mainly relies on tensorflow whilst we use sk-learn for the primitive machine learning tasks.

These are a set of simple instructions to get your environment up and running.

First you will need to install a python virtual environment using:
```
pip install virtualenv
```

Make sure you are then in the main directory of this project and run:
```
virtualenv venv
source venv/bin/activate
```
to activate the virtual environment. Once you are in this environment, you will need to install the appropriate packages by running:
```
pip install -r requirements.txt
```

Some of the code is also written in [go](https://golang.org/), which requires an [installation](https://golang.org/doc/install).
This depends on your system but if you're running macOS, I recommend using [homebrew](https://brew.sh/):
```
brew install go
```

### GPU Setup

If you plan to use the GPU support, you will also need to run some additional instructions, all of which can be found [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#test-the-tensorflow-installation) and you will need to install the GPU enabled tensorflow instead.


## Data
All of the data can be made available upon request, the reason why its not actively available is since some of the datasets are very large.

## File Structure
The project is structured as follows:
```
.
├── attacks - The source code for the existing attacks
├── data
│   └── cells - Contains all of the raw traces. Consists of a list of pairs (packetSize, 1 if outgoing else -1)
├── feature_extraction - All of the source code to extract features for different models from the raw traces
├── feature_generation - Used to automatically extract features from the raw traces
├── report - Several different reports but the most important one is the final report.
├── tests - Contains all of the unit tests
├── static - Any static resources used for either the README or the report.
├── .gitignore
├── .tavis.yml
├── README.md
└── requirements.txt
```

### References
[1] Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust scalable website fingerprinting technique." arXiv preprint arXiv:1509.00789 (2016).

[2] Panchenko, Andriy, Lukas Niessen, Andreas Zinnen, and Thomas Engel. "Website fingerprinting in onion routing based anonymization networks." In Proceedings of the 10th annual ACM workshop on Privacy in the electronic society, pp. 103-114. ACM, 2011.

[3] Panchenko, Andriy, Fabian Lanze, Andreas Zinnen, Martin Henze, Jan Pennekamp, Klaus Wehrle, and Thomas Engel. "Website fingerprinting at internet scale." In Network & Distributed System Security Symposium (NDSS). IEEE Computer Society. 2016.

[4] Gu, Xiaodan, Ming Yang, and Junzhou Luo. "A novel website fingerprinting attack against multi-tab browsing behavior." In Computer Supported Cooperative Work in Design (CSCWD), 2015 IEEE 19th International Conference on, pp. 234-239. IEEE, 2015.

[5] Wang, Tao, Xiang Cai, Rishab Nithyanand, Rob Johnson, and Ian Goldberg. "Effective Attacks and Provable Defenses for Website Fingerprinting." In USENIX Security, pp. 143-157. 2014.
