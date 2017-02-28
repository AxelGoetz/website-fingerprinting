"""
Extracts the features from the cells outlined in the paper "Effective Attacks and Provable Defenses for Website Fingerprinting" by T. Wang et al.

Then stores those features in the `../data/knn_cells` directory.
Where every file is named as follows:
`{website}-{id}.cellf`
    with both `website` and `id` being integers

The main features we are extracting are:
    - Total transmission size
    - Total transmission time
    - Numbers of incoming packets
    - Number of outgoing packets
    - Concentration of outgoing packets - Divide into 30 features and show the concentration of outgoing packets
    - Burst - sequence of outgoing packets, in which there are no two adjacent incoming packets (max, mean and number of bursts)

!! We do not include the unique packet lengths because we do not have that data readily available.
"""

from functools import reduce

def get_transmission_size_features(trace, features):
    """
    Gets the following features:
    - Transmission size
    - Total incoming/outcoming packets
    - Total transmission time
    """
    # Transmission size
    features.append(len(trace))

    # Total outgoing, total incoming
    total_outgoing = reduce(lambda acc, x: acc + 1 if x[1] > 0 else acc, trace, 0)
    total_incoming = len(trace) - total_outgoing

    features.append(total_outgoing)
    features.append(total_incoming)

    # Total transmission time
    features.append(trace[-1][0] - trace[0][0])

def get_packet_ordering(trace, features):
    """
    For each outgoing, we add a feature that indicates the total number
    of packets before it in a sequence. Also, we show the number of incoming packets between outgoing packets
    This is supposed to indicate burst patterns.

    We only go up to 300 and pad after that.
    """
    # Number of packets before it in the sequence
    count = 0
    for i, val in enumerate(trace):
        if val[1] > 0:
            count += 1
            features.append(i)
        if count == 300:
            break

    # Pad
    for i in range(count, 300):
        features.append(-1)

    # Number of incoming packets between outgoing packets
    count = 0
    prevloc = 0
    for i, val in enumerate(trace):
        if val[1] > 0:
            count += 1
            features.append(i - prevloc)
            prevloc = i
        if count == 300:
            break

    # Pad
    for i in range(count, 300):
        features.append(-1)

def concentraction_packets(trace, features):
    """
    This measure is supposed to indicate where the outgoing packets are indicated.
    We divide the trace up into non-overlapping spans of 30 packets and add the number of outgoing packets in those spans as a feature

    We only have a maximum of a 100 spans
    """
    features_added = 0
    for i in range(0, len(trace), 30):
        if i == 3000: # span_length * max_spans (30 * 100)
            break

        count = 0
        try:
            for j in range(30):
                if trace[i + j][1] > 0:
                    count += 1
        except IndexError:
            pass

        features.append(count)
        features_added += 1

    # Pad
    for i in range(0, 100 - features_added):
        features.append(0)

def bursts(trace, features):
    """
    A burst of outgoing packets is a sequence of outgoing packets where there are no two adjecent incoming packets.
    Here we find:
    - Maximum burst length
    - Mean burst length
    - Number of bursts
    - Amount of bursts greater than 5, 10, 15
    - Lengths of the first 5 bursts
    """
    bursts = []
    should_stop = 0
    current_burst_length = 0

    for i, val in enumerate(trace):
        if val[1] > 0:
            current_burst_length += 1
            should_stop = 0

        if val[1] < 0:
            if should_stop == 0:
                should_stop += 1
            elif should_stop == 1:
                bursts.append(current_burst_length)
                current_burst_length = 0
                should_stop = 0

    if current_burst_length != 0:
        bursts.append(current_burst_length)

    if len(bursts) == 0:
        features.extend([0, 0, 0, 0, 0, 0])

    else:
        features.append(max(bursts))
        features.append(sum(bursts) / len(bursts))
        features.append(len(bursts))

        counts = [0, 0, 0]
        for x in bursts:
            if x > 5:
                counts[0] += 1
            if x > 10:
                counts[1] += 1
            if x > 15:
                counts[2] += 1

        features.append(counts[0])
        features.append(counts[1])
        features.append(counts[2])

    for i in range(0, 5):
        try:
            features.append(bursts[i])
        except:
            # Pad
            features.append(-1)

def first_20_packets(trace, features):
    """
    Adds the length of the first 20 packets
    """
    for i in range(0, 20):
        try:
            features.append(trace[i][1] + 1500)
        except:
            features.append(-1)

def extract_kNN_features(trace):
    """
    Extract all of the features for the kNN model in the [kNN.py](../attacks/kNN.py) file.

    @param trace is a trace of loading a web page in the following format `[(time, incoming)]`
        where outgoing is 1 is incoming and -1
    """
    features = []

    get_transmission_size_features(trace, features)
    get_packet_ordering(trace, features)
    concentraction_packets(trace, features)
    bursts(trace, features)
    first_20_packets(trace, features)

    return features
