User Manual
===========

Firstly, in order to run the experiments, we require data. Any dataset
you want can be used, as long as each cell is represented as a Tor cell,
where the SENDMEs have been removed. However, for this specific project we will be
using two datasets, namely `GRESCHBACH`[^1] and `WANG14`[^2]. Both of
these need to be placed in the `data` directory.

After this, you need to be install and initialize a virtual environment
and install the necessary dependencies as outlined in the README.

Next, there are four main scripts that need to executed to perform the
WF attack:

1.  `train_<model_name>.py`

2.  `feature_generation.py`

3.  `feature_extraction.py`

4.  `run_models.py`

All of these scripts take several command-line parameters, which are
outlined below. These could also be displayed if you run the scripts as
follows:

``` {language="Bash"}
python <script_name>.py --help
```

Train Model
-----------

As previously mentioned, we first need to train the fingerprint
extraction model. This can be done by executing either of the commands
below in the virtual environment, depending on which model you would
like to train:

``` {language="Bash"}
python feature_generation/seq2seq/train_seq2seq.py
python feature_generation/autoencoder/train_autoencoder.py
```

Both of these take several command line parameters in order to change
the behavior of the models. All of these are outlined below:

**Parameter** | **Type** | **Default** | **Description**
--- | --- | --- | ---
batch\_size | Integer | 100 | The size of each mini-batch.
bidirectional | Boolean | False | If true, the model will use a bidirectional encoder and a normal one otherwise.
encoder\_hidden\_states | Integer | 120| The amount of hidden states in each RNN cell. The size of the fingerprints depends on this value. $\textit{len}(\textit{fingerprint}) = 2 \times \textit{encoder\_hidden\_states}$
cell\_type | String | LSTM | Which specific type of cell to use. Currently only support LSTM and GRU.
reverse\_traces | Boolean | False | If true, reverses the traces and leaves them untouched otherwise. This should not be used when `bidirectional` is true.
max\_time\_diff | Float | Infinite | The maximum time difference *(in seconds)* after which you start cutting the traces. For instance, if set to $1$, all of the traces will be cut after one second.
extension | String | .cell | The extension of the Tor cell files. We expect that they are in the following format `<webpage_id>-<instance>.<extension>`.
learning\_rate | Float | 0.000002 | The learning rate used whilst training.
batch\_norm | Boolean | False | If true, will use batch normalization within the RNN cells and otherwise the normal Tensorflow cells.

**Parameter** | **Type** | **Default** | **Description**
--- | --- | --- | ---
batch\_size | Integer | 100 | The size of each mini-batch.
extension  | String | .cell | The extension of the Tor cell files. We expect that they are in the following format `<webpage_id>-<instance>.<extension>`.
learning\_rate | Float | 0.0001 | The learning rate used whilst training.
activation\_func | String | sigmoid | The activation function used for the neurons.
layers | List | `[1500, 500, 100]` | The sizes of the respective layers in the encoder and decoder.
batch\_norm | Boolean | False | If true, will use batch normalization for the individual layers and does not perform any normalization otherwise.

  : Parameters for the `train_autoencoder.py` file.

For example to run a simple encoder-decoder with GRU cells, a batch size
of $200$ and $200$ hidden states, run the following command:

``` {language="Bash"}
python feature_generation/seq2seq/train_seq2seq.py --batch_size 200 --encoder_hidden_states 200 --cell_type "GRU"
```

After running this, a couple files should be created in the main
directory. First of all `loss_track.pkl`, which is a pickled file,
containing the object that represents the loss over time. Next, there
should also be a couple `<model_name>_model` files with different
extensions, which contain the saved computational graph. Finally, it
also creates a `X_test` and `y_test` file in the `data` directiory.
These contains the paths and the labels to the files, which were not
used for training.

Feature Generation
------------------

After the fingerprint extraction has been trained, the features need to
be extracted, which can be achieved with the `feature_generation.py`
module. Since we do not want to perform any testing on the same data as
we trained the model on, it only extracts fingerprints from the traces
in the `X_test` file.

Next, these features are stored in either the `data/seq2seq_cells` or
`data/ae_cells` directory with a `.cellf` extension.

Most of the flags here are the same as in the previous section. Hence,
we will only list the new arguments.

**Parameter** | **Type** | **Default** | **Description**
--- | --- | --- | ---
graph\_file | String | lt;model\_name|gt;\_model | The name of where you saved the graph. You should not need to change this, except if you change the graph name in the code.

  : Extra parameters for the `feature_generation.py` file.

For instance, to extract features from the model that we previously
trained, we can run:

``` {language="Bash"}
python feature_generation/seq2seq/train_seq2seq.py --batch_size 200 --encoder_hidden_states 200 --cell_type "GRU"
```

Feature Extraction
------------------

We then want to compare these automatically generated features with the
hand-picked ones. These can be extracted using the
`feature_extraction.py` script, which again has several parameters.
After this script is done running, the features should be stored in the
appropriate folders within the `data` directory. Again, all of the files
with the extracted features will have the `cellf` extension.

**Parameter** | **Type** | **Default** | **Description**
--- | --- | --- | ---
all\_files | Boolean | False | If true, it generates features for all cells and otherwise just the ones in the `X_test` file.
extension | String | .cell | Represents the extension of the cell files.

  : Parameters for the `feature_extraction.py` file.

For example:

``` {language="Bash"}
python feature_extraction.py --extension ".cells"
```

Run Models
----------

Finally, we can run the classifiers on all the extracted features using
the `run_model.py` script. After finishing the k fold validation, the
model then prints out the different scoring statistics.

**Parameter** | **Type** | **Default** | **Description**
--- | --- | --- | ---
model | String | kNN | Which model to run, the options are *kNN*, *random\_forest*, *svc1* and *svc2*.
dir\_name | String | Name of handpicked directory | If specified, it will use the features in the given directory, otherwise the standard ones for a specific model.
is\_multiclass | Boolean | True | If true, trains the classifier on a multiclass task and binary otherwise.
extension | String | .cell | Represents the extension of the cell files.

  : Parameters for the `run_model.py` file.

For example, to run a random forest model with automatically generated
fingerprints from the sequence-to-sequence model on a multiclass
problem:

``` {language="Bash"}
python run_model.py --model "random_forest" --dir_name "seq2seq_cells"
```

---

[^1]: https://nymity.ch/tor-dns/\#data

[^2]: https://cs.uwaterloo.ca/ t55wang/wf.html
