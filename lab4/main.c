#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#define alpha 100000
#define N1 256
#define N2 256
#define N3 256

double hx;
double hy;
double hz;

double x_0=-1;
double y_0=-1;
double z_0=-1;

double get_ro(double x, double y, double z)
{
	return (6-alpha*(x*x+y*y+z*z));
}

double get_new_phi(double ro, double i_prev_phi, double i_next_phi, double j_prev_phi, double j_next_phi, double k_prev_phi, double k_next_phi)
{
	double new_phi=( (i_prev_phi+i_next_phi)/(hz*hz) + (j_prev_phi+j_next_phi)/(hx*hx) +(k_prev_phi+k_next_phi)/(hy*hy) - ro  )/( 2/(hx*hx)+2/(hy*hy)+2/(hz*hz)+alpha );
	return new_phi;
}

double get_phi(double x, double y, double z)
{
	return x*x+y*y+z*z;
}

void count_inside(double *old_local_arr, double *new_local_arr, int local_n, int k1)
{
	for(int i=1;i<local_n-1;i++){
		for(int j=0;j<N1;j++){
			for(int k=0;k<N2;k++){
				if(j==0 || k==0 || j==N1-1 || k==N2-1)
				{
					new_local_arr[i*N1*N2+j*N2+k]=get_phi(x_0+j*hx, y_0+k*hy, z_0+(k1+i)*hz);
				}
				double ro=get_ro(x_0+j*hx, y_0+k*hy, z_0+(k1+i)*hz);
				double i_prev_phi=old_local_arr[(i-1)*N1*N2+j*N2+k];
				double i_next_phi=old_local_arr[(i+1)*N1*N2+j*N2+k];
				double j_prev_phi=old_local_arr[i*N1*N2+(j-1)*N2+k];
				double j_next_phi=old_local_arr[i*N1*N2+(j+1)*N2+k];
				double k_prev_phi=old_local_arr[i*N1*N2+j*N2+k-1];
				double k_next_phi=old_local_arr[i*N1*N2+j*N2+k+1];
				new_local_arr[i*N1*N2+j*N2+k]=get_new_phi(ro, i_prev_phi, i_next_phi, j_prev_phi, j_next_phi, k_prev_phi, k_next_phi);
			}
		}
	}
}

void assignment(double *old_local_arr, double *new_local_arr, int local_n)
{
	for(int i=0;i<local_n;i++){
		for(int j=0;j<N1;j++){
			for(int k=0;k<N2;k++){
				old_local_arr[i*N1*N2+j*N2+k]=new_local_arr[i*N1*N2+j*N2+k];
			}
		}
	}
}

double  get_diff(double *old_local_arr, double *new_local_arr, int local_n)
{
	double max=-1;
	for(int i=0;i<local_n;i++){
		for(int j=0;j<N1;j++){
			for(int k=0;k<N2;k++){
				double tmp=fabs(new_local_arr[i*N1*N2+j*N2+k]-old_local_arr[i*N1*N2+j*N2+k]);
				if(tmp>max)
				{
					max=tmp;
				}
			}
		}
	}
	return max;
}

void init(double *old_local_arr, int local_n, int k1)
{
	for(int i=0;i<local_n;i++){
		for(int j=0;j<N1;j++){
			for(int k=0;k<N2;k++){
				if(j==0 || k==0 || j==N1-1 || k==N2-1)
				{
					old_local_arr[i*N1*N2+j*N2+k]=get_phi(x_0+j*hx, y_0+k*hy, z_0+(k1+i)*hz);
				}
				else
				{
					old_local_arr[i*N1*N2+j*N2+k]=0;
				}
			}
		}
	}
}


int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	double t1, t2;
    t1 = MPI_Wtime();

	int dims[1]={0},periods[1]={0},coords[1],reorder=1;
	int rank,size;

	MPI_Comm comm1d;

	MPI_Comm_size(MPI_COMM_WORLD,&size);
	
	MPI_Dims_create(size,1,dims);

	size=dims[0];

	MPI_Cart_create(MPI_COMM_WORLD,1,dims,periods,reorder,&comm1d);

	MPI_Cart_get(comm1d,1,dims,periods,coords);
	rank=coords[0];

	double dx=2, dy=2, dz=2;

	double eps=0.00000001;

	hx=dx/(N1-1);
	hy=dy/(N2-1);
	hz=dz/(N3-1);

	int next, prev, neigh_phi[2], tag1=1, tag2=2;

	int k1 = N3*rank/size;
	int k2 = N3*(rank+1)/size;
	int local_n = k2 - k1;

	double *old_local_arr;
	old_local_arr=(double*)calloc(local_n*N1*N2, sizeof(double));

	double *new_local_arr;
	new_local_arr=(double*)calloc(local_n*N1*N2, sizeof(double));

	double *down;
	down=(double*)calloc(N1*N2, sizeof(double));

	double *up;
	up=(double*)calloc(N1*N2, sizeof(double));

	double *max_arr;
	max_arr=(double*)calloc(size, sizeof(double));

	/*if(rank==0)
	{
		phi=get_phi(-1, -1, -1);

	}

	if(rank==size-1)
	{
		phi=get_phi(size-1, 1, 1);

	}*/


	prev=rank-1;
	next=rank+1;

	if (rank==0)
	{
		prev=size-1;
	}
	if (rank==(size-1))
	{
		next=0;
	}

	MPI_Request reqs[4];
	MPI_Status stats[4];

	init(old_local_arr, local_n, k1);
	if(rank==0)
	{
		for(int j=0;j<N1;j++){
			for(int k=0;k<N2;k++){
				old_local_arr[j*N2+k]=get_phi(x_0+j*hx, y_0+k*hy, z_0+0*hz);
			}
		}
	}
	if(rank==size-1)
	{
		for(int j=0;j<N1;j++){
			for(int k=0;k<N2;k++){
				old_local_arr[(local_n-1)*N1*N2+j*N2+k]=get_phi(x_0+j*hx, y_0+k*hy, z_0+(local_n-1+k1)*hz);
			}
		}
	}
	int flag_to_stop=0;

	while(1)
	{
		if(rank!=0){
			for(int j=0;j<N1;j++){
				for(int k=0;k<N2;k++){
					if(j==0 || k==0 || j==N1-1 || k==N2-1)
					{
						new_local_arr[j*N2+k]=get_phi(x_0+j*hx, y_0+k*hy, z_0+0*hz);
					}
					double ro=get_ro(x_0+j*hx, y_0+k*hy, z_0+0*hz);
					double i_prev_phi=up[j*N2+k];
					double i_next_phi=old_local_arr[N1*N2+j*N2+k];
					double j_prev_phi=old_local_arr[(j-1)*N2+k];
					double j_next_phi=old_local_arr[(j+1)*N2+k];
					double k_prev_phi=old_local_arr[j*N2+k-1];
					double k_next_phi=old_local_arr[j*N2+k+1];
					new_local_arr[j*N2+k]=get_new_phi(ro, i_prev_phi, i_next_phi, j_prev_phi, j_next_phi, k_prev_phi, k_next_phi);
				}
			}
		}
		else
		{
			for(int j=0;j<N1;j++){
				for(int k=0;k<N2;k++){
					new_local_arr[j*N2+k]=get_phi(x_0+j*hx, y_0+k*hy, z_0+0*hz);
				}
			}
		}

		if(rank!=size-1){
			for(int j=0;j<N1;j++){
				for(int k=0;k<N2;k++){
					if(j==0 || k==0 || j==N1-1 || k==N2-1)
					{
						new_local_arr[(local_n-1)*N1*N2+j*N2+k]=get_phi(x_0+j*hx, y_0+k*hy, z_0+(local_n-1+k1)*hz);
					}
					double ro=get_ro(x_0+j*hx, y_0+k*hy, z_0+(local_n-1+k1)*hz);
					double i_prev_phi=old_local_arr[(local_n-2)*N1*N2+j*N2+k];
					double i_next_phi=down[j*N2+k];
					double j_prev_phi=old_local_arr[(local_n-1)*N1*N2+(j-1)*N2+k];
					double j_next_phi=old_local_arr[(local_n-1)*N1*N2+(j+1)*N2+k];
					double k_prev_phi=old_local_arr[(local_n-1)*N1*N2+j*N2+k-1];
					double k_next_phi=old_local_arr[(local_n-1)*N1*N2+j*N2+k+1];
					new_local_arr[(local_n-1)*N1*N2+j*N2+k]=get_new_phi(ro, i_prev_phi, i_next_phi, j_prev_phi, j_next_phi, k_prev_phi, k_next_phi);
				}
			}
		}
		else
		{
			for(int j=0;j<N1;j++){
				for(int k=0;k<N2;k++){
					new_local_arr[(local_n-1)*N1*N2+j*N2+k]=get_phi(x_0+j*hx, y_0+k*hy, z_0+(local_n-1+k1)*hz);
				}
			}
		}

		MPI_Irecv(up, N1*N2, MPI_DOUBLE, prev, tag1, comm1d, &reqs[0]);
		MPI_Irecv(down, N1*N2, MPI_DOUBLE, next, tag2, comm1d, &reqs[1]);

		MPI_Isend(new_local_arr, N1*N2, MPI_DOUBLE, prev, tag2, comm1d, &reqs[2]);
		MPI_Isend(&new_local_arr[(local_n-1)*N1*N2], N1*N2, MPI_DOUBLE, next, tag1, comm1d, &reqs[3]);
		//

		count_inside(old_local_arr, new_local_arr, local_n, k1);

		double max=get_diff(old_local_arr, new_local_arr, local_n);
		//printf("rank - %d , %lf\n", rank, max);
		MPI_Allgather(&max, 1, MPI_DOUBLE, max_arr, 1, MPI_DOUBLE, comm1d);

			double big_max=-1;
			for(int i=0;i<size;i++)
			{
				if(big_max<max_arr[i])
				{
					big_max=max_arr[i];
				}
			}

			if(big_max<eps)
			{
				break;
			}

		assignment(old_local_arr, new_local_arr, local_n);

		//

		MPI_Waitall(4, reqs, stats);
	}
	free(old_local_arr);

	free(new_local_arr);

	free(down);

	free(up);

	free(max_arr);

	t2 = MPI_Wtime();
	printf( "Elapsed time is %f\n", t2 - t1 );

	MPI_Finalize();

	return 0;
}
