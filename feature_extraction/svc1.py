"""
Extracts the hand-picked features used in the paper "Website Fingerprinting in Onion Routing Based Anonymization Networks" by A. Panchenko et. al.

Then stores those features in the `../data/svc1_cells` directory.
Where every file is named as follows:
`{website}-{id}.cellf`
    with both `website` and `id` being integers

Some of the features we are extracting are:
- Size markers
- Without packets that are sized 52 as they are attacks
- HTML markers - Find the size of the HTML document by counting the size of incoming packets after the first outgoing packet & the next outgoing packet
- Total Transmitted Bytes
- Percentage of incoming packets
- Number of packets

! We couldn't include occurring packet sizes
"""

def get_accumulative_list(arr):
    """
    Given an array, makes a cumulative one. Meaning that it add arr[0] to arr[1] and arr[1] to arr[2], etc.
    """
    for i in range(1, len(arr)):
        arr[i] += arr[i - 1]

    return arr

def get_size_markers(trace, features):
    """
    At each direction change of traffic, we insert a size marker that reflects how much traffic has been sent into the respective direction.
    Next, we add all of the markers reflecting incoming traffic and all the markers reflecting outcoming traffic and add those as features.

    These values are rounded and incremented by 600 (as this yields the best result).
    *(It is important that packets sized 52 (ACKs) are filtered out because otherwise traffic changes after almost every packet)*
    """
    features_added = 0
    prev = 0
    size = 0

    for val in trace:
        # Direction change
        if prev * val[1] < 0:
            if features_added >= 300:
                break

            features.append(size / 600)
            features_added += 1

        size += 1
        prev = val[1]

    for i in range(features_added, 300):
        features.append(0)

def get_html_markers(trace, features):
    """
    Finds the amount of packets received inbetween the first and second incoming packet.
    This is supposed to represent the html size.
    """
    i = 0
    count = 0
    try:
        # Find the first outgoing packet
        while trace[i][1] < 0:
            i += 1

        # Find the first incoming packet
        while trace[i][1] > 0:
            i += 1

        while trace[i][1] < 0:
            i += 1
            count += 1

    except IndexError:
        pass

    # Add 600 d a rounding element
    features.append(count + 600)

def get_total_transmitted_size(trace, features):
    """
    Adds the total amount of packets and the total amount of incoming/outcoming packets
    """
    features.append(len(trace) + 1000)

    # Outgoing
    features.append(len([x for x in trace if x[1] > 0]) + 1000)

    # Incoming
    features.append(len([x for x in trace if x[1] < 0]) + 1000)

def get_packet_stats(trace, features):
    """
    Finds the percentage of incoming packets and rounds them in steps of 5
    """
    incoming = (len([x for x in trace if x[1] < 0]) / len(trace)) * 100

    incoming = (incoming // 20) * 20

    features.append(incoming)


def extract_svc1_features(trace):
    """
    Extract all of the features for the svc model in the [svc.py](../attacks/svc.py) file.

    @param trace is a trace of loading a web page in the following format `[(time, incoming)]`
        where outgoing is 1 is incoming and -1
    """
    features = []

    get_size_markers(trace, features)
    get_html_markers(trace, features)
    get_total_transmitted_size(trace, features)
    get_packet_stats(trace, features)

    return features
