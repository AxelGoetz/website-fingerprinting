# Features
We will analyse the following hand-picked features from the following attacks:
All models appear to perform better when ACKs are removed. Hence, this will be done for all features.

## k-fingerprinting attack (Random Forest)
- Total Number of packets
- Number of incoming packets
- Number of outgoing packets
- Incoming packets as a fraction of the total packets
- Outgoing packets as a fraction of the total packets
- Packet ordering statistics // TODO: Check!!
- Divide packets into chunks of 20 and calculate standard deviation, men, median and max.
- Concentration of incoming & outgoing packets in the first and last 303 packets.
- Number of packets per second (mean, standard deviation, min, max and median)
- Divide outgoing packets into 20 chunks and sum each subset.
- Packet inter-arrival time statistics - Calculate the list of inter-arrival times between packets (extract mean, max, sd, third quartile)
- Total transmission time for incoming and outgoing packets

## SVC by A. Panchenko et al.
- Size markers (TODO: ???)
- Without packets that are sized 52 as they are attacks
- HTML markers - Find the size of the HTML document by counting the size of incoming packets after the first outgoing packet & the next outgoing packet
- Total Transmitted Bytes
- Percentage of incoming packets
- Number of packets

## Website Fingerprinting at Internet Scale - SVM with RBF kernel
They scale the features linearly to the range [-1, 1].
- Total incoming packets
- Total outgoing packets
- Sum of incoming packet sizes
- Sum of outgoing packet sizes
- Cumulative Features (see paper) n = 100

## A Novel Website Fingerprinting Attack against Multi-tab Browsing Behavior (Naive Bayes)
- Sum of incoming packet sizes
- Sum of outgoing packet sizes
- RTT (delay between first get and response)
- Size of the first packet out (GET request)
- HTML document size
- TCP Connections - we can get the number of TCP connections by counting the open channel message whose size is 96 in Linux

## Effective Attacks and Provable Defenses for Website Fingerprinting (k-NN)
- Total transmission size
- Total transmission time
- Numbers of incoming packets
- Number of outgoing packets
- Concentration of outgoing packets - Divide into 30 features and show the concentration of outgoing packets
- Burst - sequence of outgoing packets, in which there are no two adjacent incoming packets (max, mean and number of bursts)
