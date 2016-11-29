## Encoding Ordered and Unordered Attribute Sets
This software repository compliments our SOSR 2016 submission, "Concise Encoding of Flow Attributes in SDN Switches". It contains implementations of the algorithms presented in the paper, in a hopefully readable format. 

The library is broken up into two parts: a python component and a java component. The python component contains methods for generating packet tags and query strings from unordered sets or totally ordered sequences. The java component contains a single algorithm for converting a set of sequences where some sequences disagree on ordering to a set of sequences with no disagreements, where the attributes in the new sequences map to attributes in the original sequences.

In the python component, the main class is located in RSets.py. Unit tests are provided in testCases.py.

In the java component, the algorithm is located in src/TotalOrderBuilder.java. The program can be compiled using the provided makefile, and run using run.sh. The sole argument for run.sh is a filename for a file containing a list of paths. 
