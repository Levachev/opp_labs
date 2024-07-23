#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <sys/times.h>
#include <unistd.h>

void input_val(double *arr, double *b, double *first_x, int n, int local_n, int k1)
{
	for(int i=0;i<local_n;i++)
	{
		for(int j=0;j<n;j++)
		{
			arr[i*n+j]=1;
			if(k1+i==j)
			{
				arr[i*n+j]=2;
			}
		}
		b[i]=n+1;
		//first_x[k1+i]=0;
	}
}

void mul_mat_vec(double *mat, double *vec, double *result, int n, int local_n, int k1)
{
	for(int j=0;j<n;j++)
	{
		result[j]=0;
	}
	for(int i=0;i<local_n;i++)
	{
		for(int j=0;j<n;j++)
		{
			result[j]+=mat[i*n+j]*vec[i];
		}
		result[k1+i]-=(n+1);
	}
}

void def_vec(double *vec1, double *vec2, double *result, int local_n, int k1)
{
	for(int i=0;i<local_n;i++)
	{
		result[i]=vec1[i]-vec2[k1+i];
	}
}

void mul_scalar_vec(double *vec, double scalar, int local_n)
{
	for(int i=0;i<local_n;i++)
	{
		vec[i]=vec[i]*scalar;
	}
}
double get_det(double *vec, int local_n)
{
	double sum=0;
	for(int i=0;i<local_n;i++)
	{
		sum+=(vec[i]*vec[i]);
	}
	//sum=sqrt(sum);
	return sum;
}

void assigment(double *vec1, double *vec2, int local_n)
{
	for(int i=0;i<local_n;i++)
	{
		vec1[i]=vec2[i];
	}
}


int main(int argc, char *argv[]){
	

	double eps=0.00001;

	double t=0.00001;

	double det=2;

	int n=5000;

	int size, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//printf("Hello from process %d of %d\n",rank,size);

	int k1 = n*rank/size;
	int k2 = n*(rank+1)/size;
	int local_n = k2 - k1;

	double *arr;
	arr=(double*)calloc(local_n*n,sizeof(double));

	double *b;
	b=(double*)calloc(local_n, sizeof(double));

	double *first_x;
	first_x=(double*)calloc(local_n, sizeof(double));

	double *tmp_first_x;
	tmp_first_x=(double*)calloc(local_n, sizeof(double));

	double *tmp_x;
	tmp_x=(double*)calloc(n, sizeof(double));

	double *tmp_x2;
	tmp_x2=(double*)calloc(n, sizeof(double));

	int *recvcounts;
	recvcounts=(int*)calloc(size, sizeof(int));

	int *displs;
	displs=(int*)calloc(size, sizeof(int));
	
	double t1, t2; 
	t1 = MPI_Wtime();


	int sum=0;
	for(int i=0;i<size;i++)
	{
		int kk1 = n*i/size;
		int kk2 = n*(i+1)/size;
		recvcounts[i] = kk2 - kk1;
		displs[i] = sum;
		sum += recvcounts[i];
	}


	input_val(arr, b, first_x, n, local_n, k1);

	double nn = (double)(n);
	double det_b = nn*(nn+1)*(nn+1);
	det_b=sqrt(det_b);

	int iter=0;
	int iter2=0;
	int iter3=0;
	int iter4=0;

	while(1)
    {
    	MPI_Barrier(MPI_COMM_WORLD);

    	mul_mat_vec(arr, first_x, tmp_x, n, local_n, k1);

    	MPI_Allreduce(tmp_x, tmp_x2, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    	double rez_det=get_det(tmp_x2, n);

		rez_det = sqrt(rez_det);

		det=rez_det/det_b;

		//printf("%lf\n", det);
		if(det<eps)
		{
			break;
		}

		mul_scalar_vec(tmp_x2, t, n);

		def_vec(first_x, tmp_x2, tmp_first_x, local_n, k1);

		// MPI_Barrier(MPI_COMM_WORLD);

		assigment(first_x, tmp_first_x, local_n);
	}
	MPI_Allgatherv(first_x,local_n,MPI_DOUBLE, tmp_x, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

	//printf("Hello from process %d of %d\n",rank,size);
	/*for(int i=0;i<n;i++)
	{
		printf("%.9lf ", tmp_x[i]);
		//std::cout<<first_x[i]<<" ";
	}*/
	
	t2 = MPI_Wtime(); 
	printf( "Elapsed time is %f\n", t2 - t1 );
	
 	MPI_Finalize(); 
	return 0;
}
