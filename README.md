# Website Fingerprinting

// TODO: What is the project, why?

## Running the Code
Since some of the source files contain unicode characters, you need to run all of the code with `python3`.

For the rest of the installation refer to the [INSTALLATION.md](INSTALLATION.md) file.

## Data
All of the data can be made available upon request.

## File Structure
The project is structured as follows:
```
.
├── attacks - The source code for the existing attacks
├── data
│   ├── cells - Contains all of the raw traces. Consists of a list of pairs (packetSize, 1 if outgoing else -1)
│   └── knn-cells - All of the processed traces with all of the features for the kNN attack (refer to [features.md](feature-extraction/features.md) for more info).
├── feature-extraction - All of the source code to extract features for different models from the raw traces
├── feature-generation - The code to automatically generate features using deep learning
├── report - Several different reports but the most important one is the final report.
├── .gitignore
├── INSTALLATION.md
├── README.md
└── requirements.txt
```
