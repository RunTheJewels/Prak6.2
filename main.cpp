#include <cstdlib>
#include <complex>
#include <mpi.h>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <iostream>

using namespace std;

typedef std::complex<double> complexd;

void transform(complexd* a, complexd* b, const int chunk, const int k, \
               const complexd u[2][2], const int rank) {
	int K = 1 << k;
	if (K < chunk) {
		int mask = 1 << k;
		for (int i = 0; i < chunk; ++i) {
			int j;
			if ((i & mask) == 0) {
				j = i & ~mask;
				b[i] = u[0][0]*a[j];

				j = i | mask;
				b[i] += u[0][1]*a[j];
			}
			else {
				j = i & ~mask;
				b[i] = u[1][0]*a[j];

				j = i | mask;
				b[i] += u[1][1]*a[j];
			}
		}
	}
	else {
		MPI_Sendrecv(a, chunk, MPI_DOUBLE_COMPLEX, K/chunk^rank, 0, \
		             b, chunk, MPI_DOUBLE_COMPLEX, K/chunk^rank, 0, \
		             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		int mask = K / chunk;
		for (int i = 0; i < chunk; ++i)
			if ((rank & mask) == 0)
				b[i] = u[0][0]*a[i] + u[0][1]*b[i];
			else
				b[i] = u[1][1]*a[i] + u[1][0]*b[i];
	}
}

int main(int argc, char *argv[])
{
	if (argc != 3) {
		cout << "Usage: " << argv[0] << " [number of qbits]"
		     << " [number of qbit to be applied]" << endl;
		return 0;
	}
	int n = atoi(argv[1]);
	int length = 1 << n, chunk;
	int k = atoi(argv[2]);

	complexd u[2][2];

	u[0][0] = 1/sqrt(2); u[0][1] = 1/sqrt(2);
	u[1][0] = 1/sqrt(2); u[1][1] = -1/sqrt(2);

	int rank, size;

	MPI_Init(&argc,&argv);

	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	chunk = length / size;
	complexd* a = new complexd[chunk];
	complexd* b = new complexd[chunk];

	double time1 = MPI_Wtime(), start, finish, time4;
	if (READ_FROM_FILE) {
		MPI_File fh;
		MPI_File_open(MPI_COMM_WORLD,name,MPI_MODE_RDONLY,MPI_INFO_NULL,&fh);
		MPI_File_read_ordered(fh,a,chunk,MPI_DOUBLE_COMPLEX,MPI_STATUS_IGNORE);
		MPI_File_close(&fh);
	}
	else {
		double local_sum = 0, S;
		srand(time(NULL));
		unsigned int seed = (unsigned int) rand();
		MPI_Bcast(&seed,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
		seed ^= rank;
		for (int i = 0; i < chunk; ++i) {
			a[i].real() = (double) rand_r(&seed)/RAND_MAX;
			a[i].imag() = (double) rand_r(&seed)/RAND_MAX;
			// a[i] = 1.0;
			local_sum += norm(a[i]);
		}
		MPI_Allreduce(&local_sum,&S,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		S = sqrt(S);
		for (int i = 0; i < chunk; ++i)
			a[i] /= S;
		MPI_Barrier(MPI_COMM_WORLD);
	}
	start = MPI_Wtime();
	transform(a,b,chunk,n-k,u,rank);
	MPI_Barrier(MPI_COMM_WORLD);
	finish = MPI_Wtime();

	delete[] a;
	delete[] b;

	if (rank == 0) {
		cout << n << ' ' << size << ' ' << finish-start << endl;
	}

	MPI_Finalize();
	return 0;
}
