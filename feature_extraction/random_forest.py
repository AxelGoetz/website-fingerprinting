"""
Extracts the hand-picked features used in the paper "k-fingerprinting: a Robust Scalable Website Fingerprinting Technique" by J. Hayes et. al.

Then stores those features in the `../data/rf_cells` directory.
Where every file is named as follows:
`{website}-{id}.cellf`
    with both `website` and `id` being integers

Some of the features we are extracting are:
- Total Number of packets
- Number of incoming packets
- Number of outgoing packets
- Incoming packets as a fraction of the total packets
- Outgoing packets as a fraction of the total packets
- Packet ordering statistics
- Divide packets into chunks of 20 and calculate standard deviation, men, median and max.
- Concentration of incoming & outgoing packets in the first and last 303 packets.
- Number of packets per second (mean, standard deviation, min, max and median)
- Divide outgoing packets into 20 chunks and sum each subset.
- Packet inter-arrival time statistics - Calculate the list of inter-arrival times between packets (extract mean, max, sd, third quartile)
- Total transmission time for incoming and outgoing packets
"""

from functools import reduce
from math import floor, ceil

import numpy as np

# Helpers
# -----------------------------------------

def counts_in_out_packets(packets):
    """
    Counts the number of packets in & out in the array packets

    @param packets is a list of packets, structured as follows: `[(time, direction)]`

    @return tuple `(num_packets_in, num_packets_out)`
    """
    packets_in, packets_out = [], []

    for val in packets:
        if val[1] < 0:
            packets_in.append(val)
        elif val[1] > 0:
            packets_out.append(val)

    return (len(packets_in), len(packets_out))

def inter_packet_time(trace):
    """
    For a trace, returns a list of floats that represent the time difference between two adjecent packets.

    @param trace is a list structered as follows: `[(time, direction)]`
    """
    times = []

    for elem, next_elem in zip(trace[:-1], trace[1:]):
        times.append(next_elem[0] - elem[0])

    return times

# Feature extraction
# -----------------------------------------

def get_number_packets_stats(trace, features):
    """
    Gets statistics regarding the packets:
    - Total number of packets
    - Number of incoming and outgoing packets
    - Number of incoming & outgoing packeta as a fraction of total packets
    """
    # Transmission size
    features.append(len(trace))

    # Total outgoing, total incoming
    total_outgoing = reduce(lambda acc, x: acc + 1 if x[1] > 0 else acc, trace, 0)
    total_incoming = len(trace) - total_outgoing

    features.append(total_outgoing)
    features.append(total_incoming)

    features.append(total_outgoing / len(trace))
    features.append(total_incoming / len(trace))

def get_packet_ordering(trace, features):
    """
    For outgoing and incoming packets, the number of packet is has seen before that in the sequence.
    We want:
    - Average of both incoming and outgoing
    - Standard deviation
    """
    incoming, outgoing = [], []
    for i, val in enumerate(trace):
        if val[1] > 0:
            outgoing.append(i)
        elif val[1] < 0:
            incoming.append(i)

    # Average
    features.append(sum(incoming) / len(incoming))
    features.append(sum(outgoing) / len(outgoing))

    # Standard deviation
    features.append(np.std(incoming))
    features.append(np.std(outgoing))

def get_packet_concentration(trace, features):
    """
    Packet sequences are split into non-overlapping chunks of 20 packets.
    We count the number of outgoing packets in each of the chunks.
    Next, we extract, standard deviation, mean, median and max.

    This can be used to provide a snapshot of where outgoing pacets are concentrated.

    @return a list of integers, each representing how many outgoing packets there are in 20 packet-sized chucks
    """
    chunks = [trace[x:x+20] for x in range(0, len(trace), 20)]

    concentrations = []

    for item in chunks:
        c = 0
        for p in item:
            if p[1] > 0:
                c += 1
        concentrations.append(c)

    features.append(np.std(concentrations))
    features.append(sum(concentrations) / len(concentrations))
    features.append(np.percentile(concentrations, 50))
    features.append(max(concentrations))

    return concentrations

def get_number_packets_start_end(trace, features):
    """
    Gets the number of incoming & outcoming packets in the first and last 30 packets
    """
    first = trace[:30]
    last = trace[-30:]

    packets_in, packets_out = counts_in_out_packets(first)

    features.append(packets_in)
    features.append(packets_out)

    packets_in, packets_out = counts_in_out_packets(last)

    features.append(packets_in)
    features.append(packets_out)

def get_packets_per_second(trace, features):
    """
    Gets the total number of packets per second along with mean, standard deviation, min, max and median

    @return a 1D list that contains the packets per second
    """
    packets_per_second = {}
    for val in trace:
        second = floor(val[0])

        if second not in packets_per_second:
            packets_per_second[second] = 0

        packets_per_second[second] += 1

    l = list(packets_per_second.values())

    features.append(sum(l) / len(l))
    features.append(np.std(l))
    features.append(np.percentile(l, 50))
    features.append(min(l))
    features.append(max(l))

    return l

def get_alternative_concentration(concentration, features):
    """
    Divide the concentration array up into 20 evenly sized subsets and sum each subset.
    These 20 values are now a list of new features.

    @param concentration a list of integers, each representing how many outgoing packets there are in 20 packet-sized chucks
    """

    chunk_length = ceil(len(concentration) / 20)
    chunks = []

    if chunk_length == 0:
        chunks = concentration
    else:
        chunks = [sum(concentration[x:x+chunk_length]) for x in range(0, len(concentration), chunk_length)]

    chunks.extend([0] * (20 - len(chunks)))

    for chunk in chunks:
        features.append(chunk)

    features.append(sum(chunks))

def get_inter_arrival_time(trace, in_trace, out_trace, features):
    """
    For both the complete trace, the trace only containing incoming and the trace only containing outcoming packets,
    we calculate the inter-arrival time.
    Next, we add the max, mean, standard deviation and third quartile as features.

    @param in_trace contains all the incoming packets and nothing else
    @param out_trace contains all the outcoming packets and nothing else
    """
    complete_inter_arrival_time = inter_packet_time(trace)

    in_inter_arrival_time = inter_packet_time(in_trace)
    out_inter_arrival_time = inter_packet_time(out_trace)

    inter_arrival_times = [complete_inter_arrival_time, in_inter_arrival_time, out_inter_arrival_time]

    for time in inter_arrival_times:
        if len(time) == 0:
            features.extend([0, 0, 0, 0])

        else:
            features.append(max(time))
            features.append(sum(time) / len(time))
            features.append(np.std(time))
            features.append(np.percentile(time, 75))

def get_transmission_time_stats(trace, in_trace, out_trace, features):
    """
    For the complete trace and the traces only containing incoming and outcoming packets, we extract the following:
    - First, second and third quartile
    - Total transmission time
    """
    in_times    = [x[0] for x in in_trace]
    out_times   = [x[0] for x in out_trace]
    total_times = [x[0] for x in trace]

    times = [total_times, in_times, out_times]

    for time in times:
        if len(time) == 0:
            features.extend([0, 0, 0, 0])

        else:
            features.append(np.percentile(time, 25))
            features.append(np.percentile(time, 50))
            features.append(np.percentile(time, 75))
            features.append(np.percentile(time, 100))

def extract_rf_features(trace):
    """
    Extract all of the features for the kFingerprinting model in the [random_forest.py](../attacks/random_forest.py.py) file.

    @param trace is a trace of loading a web page in the following format `[(time, incoming)]`
        where outgoing is 1 is incoming and -1
    """
    features = []

    get_number_packets_stats(trace, features)
    get_packet_ordering(trace, features)
    concentration = get_packet_concentration(trace, features)
    get_number_packets_start_end(trace, features)
    packets_per_sec = get_packets_per_second(trace, features)

    in_trace  = [x for x in trace if x[1] < 0]
    out_trace = [x for x in trace if x[1] > 0]

    get_alternative_concentration(concentration, features)
    get_alternative_concentration(packets_per_sec, features)
    get_inter_arrival_time(trace, in_trace, out_trace, features)
    get_transmission_time_stats(trace, in_trace, out_trace, features)

    return features
