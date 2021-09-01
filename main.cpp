#include <iostream>
#include <vector>
#include <mpi.h>
#include <algorithm>
#include <string>
#include <sstream>
#include <unistd.h>
#include <cassert>
#include <climits>

void parallelRange(MPI_Comm comm, int commStart, int commStop, int &localStart, int &localStop)
{
	int rank, nproc;
	int globalSize;
	int localSize;
	int localRemainder;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	globalSize = commStop - commStart;

	localSize = globalSize / nproc;

	localRemainder = globalSize % nproc;

	if (localRemainder > rank)
		localSize++;

	MPI_Scan(&localSize, &localStop, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	localStart = localStop - localSize;

	localStart += commStart;
	localStop += commStart;
}

void printArray(int *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		std::cout << array[i] << " ";
	}
	std::cout << std::endl;
}

void printArray(std::vector<int> array)
{
	printArray(&array[0], array.size());
}

void scatterArray(std::vector<int> array, std::vector<int> &scattered)
{
	int rank, nproc;
	int size = 0;
	int globalSize = 0;
	int offset;
	int localStart, localStop, localSize;

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int sendcounts[nproc];
	int displs[nproc];

	globalSize = array.size();
	MPI_Bcast(&globalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

	parallelRange(MPI_COMM_WORLD, 0, globalSize, localStart, localStop);

	localSize = localStop - localStart;

	MPI_Gather(&localSize, 1, MPI_INT, &sendcounts[0], 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&localStart, 1, MPI_INT, &displs[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

	scattered.resize(localSize);

	MPI_Scatterv(&array[0], &sendcounts[0], &displs[0], MPI_INT, &scattered[0], localSize, MPI_INT, 0, MPI_COMM_WORLD);
}

void gatherArray(std::vector<int> array, std::vector<int> &gathered)
{
	int rank, nproc;
	int size = 0;
	int globalSize = 0;
	int offset;

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int recvcounts[nproc];
	int displs[nproc];

	size = array.size();
	MPI_Reduce(&size, &globalSize, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	gathered.resize(globalSize);

	MPI_Scan(&size, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	offset -= size;

	MPI_Gather(&size, 1, MPI_INT, &recvcounts[0], 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&offset, 1, MPI_INT, &displs[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Gatherv(&array[0], size, MPI_INT, &gathered[0], recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
}

void mergeArrays(MPI_Comm comm, std::vector<int> &array)
{
	int rank, nproc;
	MPI_Comm_size(comm, &nproc);
	MPI_Comm_rank(comm, &rank);

	std::vector<int> tomerge;
	int recvcounts[nproc];
	int displs[nproc];
	int size, offset, commSize;

	size = array.size();
	MPI_Scan(&size, &offset, 1, MPI_INT, MPI_SUM, comm);
	offset -= size;

	commSize = 0;
	MPI_Reduce(&size, &commSize, 1, MPI_INT, MPI_SUM, 0, comm);
	tomerge.resize(commSize);

	MPI_Gather(&size, 1, MPI_INT, &recvcounts[0], 1, MPI_INT, 0, comm);
	MPI_Gather(&offset, 1, MPI_INT, &displs[0], 1, MPI_INT, 0, comm);
	MPI_Gatherv(&array[0], size, MPI_INT, &tomerge[0], recvcounts, displs, MPI_INT, 0, comm);

	//merge
	std::vector<int> copy(tomerge.size());
	if (rank == 0)
	{
		std::cout << "merging result from " << nproc << " processes." << std::endl;
		std::vector<int> starts(displs, displs + nproc);
		std::vector<int> stops(nproc);

		for (int iproc = 0; iproc < nproc; iproc++)
		{
			stops[iproc] = starts[iproc] + recvcounts[iproc];
		}

		int ptr = 0;
		while (ptr < tomerge.size())
		{
			int min = INT_MAX;
			int hasmin = -1;
			for (int iproc = 0; iproc < nproc; iproc++)
			{
				if (starts[iproc] < stops[iproc] && min > tomerge[starts[iproc]])
				{
					min = tomerge[starts[iproc]];
					hasmin = iproc;
				}
			}
			assert(hasmin != -1);
			copy[ptr++] = tomerge[starts[hasmin]++];
		}
	}

	MPI_Scatterv(&copy[0], recvcounts, displs, MPI_INT, &array[0], size, MPI_INT, 0, comm);
}

void parallelSort(MPI_Comm comm, std::vector<int> &array)
{
	MPI_Comm split;
	int rank, nproc;
	int color, key;

	MPI_Comm_size(comm, &nproc);
	MPI_Comm_rank(comm, &rank);

	if (nproc > 1)
	{
		if (rank < nproc / 2)
			color = 0;
		else
			color = 1;
		key = rank;

		MPI_Comm_split(comm, color, key, &split);

		parallelSort(split, array);
		mergeArrays(comm, array);

		MPI_Comm_free(&split);
	}
	else
	{
		std::sort(array.begin(), array.end());
	}
}

bool checkSorted(std::vector<int> array)
{
	bool result = true;
	int rank, nproc;
	int size = array.size();

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	for (int i = 1; i < size; i++)
	{
		if (array[i] < array[i - 1])
		{
			result = false;
		}
	}

	int starts[nproc];
	int stops[nproc];

	MPI_Gather(&array[0], 1, MPI_INT, &starts[0], 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&array[size - 1], 1, MPI_INT, &stops[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 1; i < nproc; i++)
	{
		if (stops[i] < starts[i - 1])
		{
			result = false;
		}
	}

	return result;
}

int main(int argc, char **argv)
{
	int rank, nproc;
	int arraySize;
	std::vector<int> init(0), unsorted(0);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	arraySize = atoi(argv[1]);

	if (rank == 0)
	{
		init.resize(arraySize);

		for (auto &v : init)
		{
			v = rand() % (2 * arraySize);
		}
	}

	scatterArray(init, unsorted);

	double time = MPI_Wtime();
	parallelSort(MPI_COMM_WORLD, unsorted);

	bool isSorted = checkSorted(unsorted);

	time = MPI_Wtime() - time;

	if (rank == 0)
	{
		std::cout << "Time to sort: " << time << " s" << std::endl;
		std::cout << "Array is sorted: " << isSorted << std::endl;
	}

	MPI_Finalize();
}