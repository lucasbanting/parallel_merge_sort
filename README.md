# parallel_merge_sort

Implementation of merge sort with MPI. 
Merges arrays on a single process so algorithm has same complexity as serial merge sort algorithm.'

To build use:
  $ make

To run use:

  $ mpirun -n np main.out N
  
Where np is number of processors to use and N is number of elements to sort.
