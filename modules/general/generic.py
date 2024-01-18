"""
A collection of generic functions that needn't be customised.

did_fire - screen neurons to eliminate those that don't fire close to the
    earliest time.

find_small_qualified - part of a sorting algorithm intended to find
    the smallest value in an array, except it only pays attention to
    elements with a raised flag.
    Finds the smallest element in each block then returns it in a
    smaller array.

find_smallest - non-CUDA wrapper for find_small_qualified that also
    finishes the job of sorting the returned smaller array.
    Being host-side means you could set it up as an entirely host-side
    sort if that turns out to be faster.
"""

from numba import cuda
from math import e
import numpy

from modules.general import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
leniency_threshold = lookup["leniency_threshold"]
v_r = lookup["v_r"]
synapse_decay = lookup["synapse_decay"]

threads = lookup["threads"]
blocks = lookup["blocks"]



@cuda.jit()
def cull_larger(some_array, flags, threshold):
    n = cuda.grid(1)
    if n < neurons_number:
        #flags[n] *= (some_array[n] < threshold)
        if some_array[n] > threshold:
            flags[n] = 0
        
@cuda.jit()
def find_small_qualified(some_array, flags, results_array, fill):
    """
    Filters an array for small elements and stores them in a smaller array.
    Only looks at elements with a flag.
    One thread per block does all the work, and the reduction factor is the
    number of threads per block.
    Returns an erroneous result if all flags are false, but shouldn't matter.
    """
    n = cuda.grid(1)
    rank = cuda.threadIdx.x
    if (n < neurons_number) and (rank == 0):
        limit = cuda.libdevice.min(threads, neurons_number - n)
        lowest = fill
        lowest_flag = False
        for m in range(n + 1, n + limit):
            if flags[m]:
                if lowest_flag == False:
                    lowest = some_array[m]
                    lowest_flag = True
                elif some_array[m] < lowest:
                    lowest = some_array[m]
##                    lowest = lowest * (lowest < some_array[m]) \
##                             + some_array[m] * (some_array[m] < lowest)
        results_array[n // threads] = lowest
        
def find_smallest(some_device_list, flags, partial_device_list, partial_list):
    """
    Finds the smallest element in an array with the help of kernel calls.
    Only considers elements with a raised flag to be valid.
    Wants all the 1D arrays provided to it already, and doesn't do any
    checks on lengths.    
    """
    fill = numpy.ma.minimum_fill_value(partial_device_list)
    
    find_small_qualified[blocks, threads](some_device_list, flags,
                                          partial_device_list, fill)
    partial_device_list.copy_to_host(partial_list)
    #print(partial_list)

    lowest = partial_list[0]
    for i in range(1, len(partial_list)):
        if partial_list[i] < lowest:
            lowest = partial_list[i]
    return lowest


if __name__ == "__main__":
    #Test finding the smallest element
    from numpy.random import default_rng
    length = 1001
    
    random_array = default_rng().random(length)
    #print(random_array)
    numpy_least = numpy.amin(random_array)
    flag_array = numpy.ones(length)
    sub_array = numpy.zeros(blocks)
    d_random_array = cuda.to_device(random_array)
    d_flag_array = cuda.to_device(flag_array)
    d_sub_array = cuda.to_device(sub_array)
    my_least = find_smallest(d_random_array, d_flag_array,
                             d_sub_array, sub_array)

    print("Numpy's least value:", numpy_least)
    print("My least value:", my_least)
