# One of the first steps in designing a parallel program is to break the problem into discrete “chunks” of work that can be distributed to multiple tasks so the can work on on the problem simultaneously.
# This is known as decomposition or partitioning. There are two main ways to decompose an algorithm: domain decomposition and functional decomposition.
# An example of domain decomposition can be seen by computing a simple integral using the Mid-point rule.

# Parallel point-to-point version

# Since the problem has already been decomposed into separate partitions, it is easy to implement a parallel version of the algorithm. In this case, each of the partitions can be computed by a separate process. 
# Once each process has computed its partition, it sends the result back to a root process (in this case process 0) which sums the values and prints the final result


import numpy
from math import acos, cos
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def integral(a_i, h, n):
    integ = 0.0
    for j in range(n):
        a_ij = a_i + (j + 0.5) * h
        integ += cos(a_ij) * h
    return integ

pi = 3.14159265359
n = 500
a = 0.0
b = pi / 2.0
h = (b - a) / (n * size)
a_i = a + rank * n * h

# All processes initialize my_int with their partition calculation
my_int = numpy.full(1, integral(a_i, h, n))

print("Process ", rank, " has the partial integral ", my_int[0])

if rank == 0:
    # Process 0 receives all the partitions and computes the sum
    integral_sum = my_int[0]
    for p in range(1, size):
        comm.Recv(my_int, source=p)
        integral_sum += my_int[0]

    print("The integral = ", integral_sum)
else:
    # All other processes just send their partition values to process 0
    comm.Send(my_int, dest=0)