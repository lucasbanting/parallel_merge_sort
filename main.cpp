#include <iostream>
#include <vector>
#include <mpi.h>
#include <algorithm>
#include <string>
#include <sstream>
#include <unistd.h>
#include <cassert>
#include <climits>

#define DISTRIBUTE_BUCKET 1

void parallelRange(MPI_Comm comm, int commStart, int commStop, int &localStart, int &localStop)
{
	int rank, nproc;
	int globalSize;
	int localSize;
	int localRemainder;
	int offset;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	globalSize = commStop - commStart;
	localSize = globalSize / nproc;
	localRemainder = globalSize % nproc;

	if(rank < localRemainder)
		offset = rank;
	else
		offset = localRemainder;

	localStart = rank*localSize + offset;
	localStop = localStart + localSize;
	if(localRemainder > rank) localStop++;

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
		//std::cout << "merging result from " << nproc << " processes." << std::endl;
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

int binaryBucketSearch(std::vector<int> splitters, int low, int high, int val)
{
	int mid;
	int bucket = -1;
	while(low < high -1)
	{	
		mid = (low+high)/2;
		
		if(val < splitters[mid])
		{
			high = mid;
		}	
		else
		{
			low = mid;
		}
	}
	
	if( val <= splitters[mid])
		bucket = mid-1;
	else
		bucket = mid;

	return bucket;
}

//************************************************************************
// use immediate sends and recieves to send correct bucket to each process
//************************************************************************	
void distributeBuckets(MPI_Comm comm, std::vector<std::vector<int>> buckets, std::vector<int> &array)
{	
	int rank, nproc;
	
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	std::vector<MPI_Request> requests(2*nproc);
	std::vector<int> recvcounts(nproc);
	std::vector<int> recvdispls(nproc);
	int totalRecv=0;

	for(int ibucket=0; ibucket < buckets.size(); ibucket++)
	{
		MPI_Isend(&buckets[ibucket][0], buckets[ibucket].size(), MPI_INT, ibucket, DISTRIBUTE_BUCKET, comm, &requests[ibucket]);
	}

	for(int ibucket=0; ibucket < buckets.size(); ibucket++)
	{
		MPI_Status status;
		MPI_Probe(ibucket, DISTRIBUTE_BUCKET, comm, &status);
		MPI_Get_count(&status, MPI_INT, &recvcounts[ibucket]);
	}

	recvdispls[0]=0;
	for(int ibucket=1; ibucket < buckets.size(); ibucket++)
	{
		recvdispls[ibucket] = recvdispls[ibucket-1] + recvcounts[ibucket-1];
	}
	totalRecv = recvcounts[nproc-1] + recvdispls[nproc-1];

	array.resize(totalRecv);
	for(int ibucket=0; ibucket < buckets.size(); ibucket++)
	{
		MPI_Irecv(&array[recvdispls[ibucket]], recvcounts[ibucket], MPI_INT, ibucket, DISTRIBUTE_BUCKET, comm, &requests[nproc + ibucket]);
	}
	
	MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
}

void sampleSort(MPI_Comm comm, std::vector<int> &array, int k)
{
	int rank, nproc;
	int size;
	std::vector<int> samples, allsamples, splitters;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	//****************************
	// sample array to find pivots
	//****************************
	size=array.size();
	
	for(int isample=0; isample < k; isample++)
	{
		samples.push_back(array[size*isample/k]);
	}

	allsamples.resize(nproc*k);
	MPI_Allgather(&samples[0], k, MPI_INT, &allsamples[0], k, MPI_INT, comm);

	//**********************************************
	// pick evenly spaced pivots from sorted samples
	//**********************************************
	sort(allsamples.begin(), allsamples.end());

	splitters.push_back(INT_MIN);
	for(int isplitter=0; isplitter < nproc-1; isplitter++)
	{
		splitters.push_back(allsamples[k*isplitter + k]);
	}
	splitters.push_back(INT_MAX);
	

	//********************************
	// place array values into buckets
	//********************************
	std::vector<std::vector<int>> buckets(nproc);
	for(auto val : array)
	{
		int bucket = binaryBucketSearch(splitters, 0, splitters.size()-1, val);
		buckets[bucket].push_back(val);
	}

	distributeBuckets(comm, buckets, array);
	

	//*********************
	// sort the local array
	//*********************
	sort(array.begin(), array.end());


	//*****************************
	// redistribute/rebalance array
	//*****************************
	int totalSize = array.size();
	int actualStart;
	int localStart, localStop, localSize;
	std::vector<int> localStarts(nproc);

	MPI_Scan(&totalSize, &actualStart, 1, MPI_INT, MPI_SUM, comm);
	actualStart -= totalSize;

	MPI_Allreduce(MPI_IN_PLACE, &totalSize, 1, MPI_INT, MPI_SUM, comm);

	parallelRange(comm, 0, totalSize, localStart, localStop);
	localSize = localStop - localStart;

	MPI_Allgather(&localStart, 1, MPI_INT, &localStarts[0], 1, MPI_INT, comm);
	localStarts.push_back(INT_MAX);

	for(int ibucket=0; ibucket < nproc; ibucket++)
	{
		buckets[ibucket].resize(0);
	}

	for(int ival=0; ival < array.size(); ival++)
	{	
		int low = 0;
		int high = localStarts.size()-1;
		int mid = -1;
		int key = actualStart + ival;

		while(low < high)
		{
			mid = (low + high)/2;
			if(key >= localStarts[mid] && key < localStarts[mid+1])
			{
				break;
			}
			else if(key >= localStarts[mid])
			{
				low = mid;
			}
			else
			{
				high = mid;
			}
		}
		buckets[mid].push_back(array[ival]);
	}

	
	distributeBuckets(comm, buckets, array);

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
		std::cout << "Time to merge-sort: " << time << " s" << std::endl;
		std::cout << "Array is sorted: " << isSorted << std::endl;
	}

	// reset array
	scatterArray(init, unsorted);


	time = MPI_Wtime();
	sampleSort(MPI_COMM_WORLD, unsorted, 64);

	isSorted = checkSorted(unsorted);

	time = MPI_Wtime() - time;

	if (rank == 0)
	{
		std::cout << "Time to sample-sort: " << time << " s" << std::endl;
		std::cout << "Array is sorted: " << isSorted << std::endl;
	}

	MPI_Finalize();
}