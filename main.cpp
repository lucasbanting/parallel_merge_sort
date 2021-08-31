#include <iostream>
#include <vector>
#include <mpi.h>
#include <algorithm>

int main(int argc, char **argv)
{
	int rank, nproc;
	int arraySize;
	std::vector<int> unsorted, sorted;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	arraySize = atoi(argv[1]);
	

	MPI_Finalize();
}