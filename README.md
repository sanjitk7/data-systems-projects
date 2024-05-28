# Projects at the intersection of Machine Learning and Data Systems

This repository contains projects undertaken during the graduate class ML and Data Systems which focuses on techniques and research at the intersection of the two fields.


## Project 1: Online Aggregation (OLA) for basic Pandas Operations and HyperLogLog Cardinality

Online aggregation in data systems refers to a technique that allows users to obtain approximate query results progressively, providing incremental and continually improving estimates as the computation proceeds. Instead of waiting for the entire query to be processed, users can see early estimates and refine them over time as more data is processed. This approach is particularly useful in interactive data exploration, where quick insights are needed, and for long-running queries, enabling users to make decisions based on preliminary results while the final computation is still underway. Online aggregation improves user experience by providing timely, albeit approximate, feedback and allows for early termination if the intermediate results are sufficiently accurate.

Here OLA is implemented for some basic Python Pandas DF operations:

Filtered mean, i.e., avg(x) where y = z 
Grouped means, i.e., avg(x) group by y 
Grouped sums, i.e., sum(x) group by y 
Grouped counts, i.e., count(x) group by y 
Filtered cardinality via HyperLogLog , i.e., count_distinct(x) where y = z 

## Project 2: Apache Arrow ASCII Encoder Decoder

The subfolder is a fork of the original Apache Arrow project. For the specific work done check [apache-arrow-parquet-cpp-encode-decode/cpp/cs598/ascii/README.md](apache-arrow-parquet-cpp-encode-decode/cpp/cs598/ascii/README.md)

Modified the Apache Arrow source code to implement a new ASCII Encoder and Decoder for Parquet, supporting the Integer and Float data types.



## Project 3: Flatbuffers

Worked with Google's [Flatbuffers](https://github.com/google/flatbuffers) and Python's [shared memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html) libraries to pass serialized DataFrames between notebook sessions, and performed various operations (`head`, `groupby`, `map`) directly on the serialized DataFrames. This improves data sharing and manipulation efficiency in Python since leveraging Flatbuffers provides faster serialization and shared memory for inter-process communication.