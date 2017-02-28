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
- Cumulative Features (see paper) n = 100
"""

def get_packet_stats(trace, features):
    """
    Get packet stats such as total, incoming and outcoming packets
    """
    features.append(len(trace))

    # Outgoing
    features.append(len([x for x in trace if x[1] > 0]))

    # incoming
    features.append(len([x for x in trace if x[1] < 0]))


def get_cumulative_representation(trace, features, n):
    """
    Gets a cumulative representation of a trace, described in the "Website Fingerprinting at Internet Scale" paper.

    @param n is the amount of features to be added.
        It affects how often and the places where you sample
    """
    a, c = 0, 0

    sample = (len(trace) // n)
    amount = 0

    for i, packet in enumerate(trace):
        c += packet[1]
        a += abs(packet[1])

        if i % sample == 0:
            amount += 1
            features.append(c)
            features.append(a)

            if amount == n:
                break

    for i in range(amount, n):
        features.append(0)
        features.append(0)


def extract_svc2_features(trace):
    """
    Extract all of the features for the svc model in the [svc.py](../attacks/svc.py) file.

    @param trace is a trace of loading a web page in the following format `[(time, incoming)]`
        where outgoing is 1 is incoming and -1
    """
    features = []

    get_packet_stats(trace, features)
    # n = 100 yields the best result
    get_cumulative_representation(trace, features, 100)

    return features
