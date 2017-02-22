"""
Extracts the hand-picked features used in the paper "Website Fingerprinting at Internet Scale" by A. Panchenko et. al.

Then stores those features in the `../data/svc2_cells` directory.
Where every file is named as follows:
`{website}-{id}.cellf`
    with both `website` and `id` being integers

They scale the features linearly to the range [-1, 1].

Some of the features we are extracting are:
- Total incoming packets
- Total outgoing packets
- Sum of incoming packet sizes
- Sum of outgoing packet sizes
- Cumulative Features (see paper) n = 100
"""

def extract_svc2_features(trace):
    """
    Extract all of the features for the svc model in the [svc.py](../attacks/svc.py) file.

    @param trace is a trace of loading a web page in the following format `[(time, incoming)]`
        where outgoing is 1 is incoming and -1
    """
    pass
