"""
Extracts the hand-picked features used in the paper "A novel website fingerprinting attack against multi-tab browsing behavior." by X. Gu et. al.

Then stores those features in the `../data/nb_cells` directory.
Where every file is named as follows:
`{website}-{id}.cellf`
    with both `website` and `id` being integers

Some of the features we are extracting are:
- Sum of incoming packet sizes
- Sum of outgoing packet sizes
- RTT (delay between first get and response)
- HTML document size
"""

import numpy as np

def inter_packet_time(trace):
    """
    For a trace, returns a list of floats that represent the time difference between two adjecent packets.

    @param trace is a list structered as follows: `[(time, direction)]`
    """
    times = []

    # From start to -1 and from 1 to end
    for elem, next_elem in zip(trace[:-1], trace[1:]):
        times.append(next_elem[0] - elem[0])

    return times

def sum_in_out_packets(trace, features):
    """Calculates the amount of incoming and outcoming packets"""
    packets_in, packets_out = [], []

    for val in trace:
        if val[1] < 0:
            packets_in.append(val)
        elif val[1] > 0:
            packets_out.append(val)

    features.append(len(packets_in))
    features.append(len(packets_out))

def get_rtt(trace, features):
    """
    Calculates the round trip time (rtt) by finding the time between the first get packet and the first response packet
    """
    i = 0
    first_outgoing_packet = -1
    first_incoming_packet = -1

    try:
        while i < len(trace):
            if trace[i][1] > 0:
                first_outgoing_packet = trace[i][0]
                break
            i += 1

        while i < len(trace):
            if trace[i][1] < 0:
                first_incoming_packet = trace[i][0]
                break
            i += 1

    except IndexError:
        pass

    features.append(first_incoming_packet - first_outgoing_packet)

def get_html_size(trace, features):
    """
    Finds the amount of packets received inbetween the first and last incoming packet of the first burst.
    This is supposed to represent the html size.
    """
    i = 0
    count = 1
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

    features.append(count)

def get_inter_arrival_time(trace, features):
    """
    For both the complete trace, the trace only containing incoming and the trace only containing outcoming packets,
    we calculate the inter-arrival time.
    Next, we add the max, mean and standard deviation as features.
    """
    complete_inter_arrival_time = inter_packet_time(trace)

    in_trace  = [x for x in trace if x[1] < 0]
    out_trace = [x for x in trace if x[1] > 0]

    in_inter_arrival_time = inter_packet_time(in_trace)
    out_inter_arrival_time = inter_packet_time(out_trace)

    inter_arrival_times = [complete_inter_arrival_time, in_inter_arrival_time, out_inter_arrival_time]

    for time in inter_arrival_times:
        if len(time) == 0:
            features.extend([0, 0, 0])

        else:
            features.append(max(time))
            features.append(sum(time) / len(time))
            features.append(np.std(time))


def extract_nb_features(trace):
    """
    Extract all of the features for the naive_bayes model in the [naive_bayes.py](../attacks/naive_bayes.py) file.

    @param trace is a trace of loading a web page in the following format `[(time, incoming)]`
        where outgoing is 1 is incoming and -1
    """
    features = []

    sum_in_out_packets(trace, features)
    get_rtt(trace, features)
    get_html_size(trace, features)
    get_inter_arrival_time(trace, features)

    return features
