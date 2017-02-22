"""
Extracts the hand-picked features used in the paper "Website Fingerprinting in Onion Routing Based Anonymization Networks" by A. Panchenko et. al.

Then stores those features in the `../data/svc1_cells` directory.
Where every file is named as follows:
`{website}-{id}.cellf`
    with both `website` and `id` being integers

Some of the features we are extracting are:
- Size markers (TODO: ???)
- Without packets that are sized 52 as they are attacks
- HTML markers - Find the size of the HTML document by counting the size of incoming packets after the first outgoing packet & the next outgoing packet
- Total Transmitted Bytes
- Percentage of incoming packets
- Number of packets
"""

def extract_svc1_features(trace):
    """
    Extract all of the features for the svc model in the [svc.py](../attacks/svc.py) file.

    @param trace is a trace of loading a web page in the following format `[(time, incoming)]`
        where outgoing is 1 is incoming and -1
    """
    pass
