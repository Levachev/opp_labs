#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <iostream>
#include <omp.h>

int n;
double sum=0;

void input_val(double *arr, double *b, double *first_x)
{
	#pragma omp for
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			arr[i*n+j]=1;
			if(i==j)
			{
				arr[i*n+j]=2;
			}
		}
		b[i]=n+1;
		first_x[i]=0;
	}
}

void mul_mat_vec(double *mat, double *vec, double *result)
{
	#pragma omp for
	for(int i=0;i<n;i++)
	{
		result[i]=0;
		for(int j=0;j<n;j++)
		{
			result[i]+=mat[i*n+j]*vec[j];
		}
	}
}

void def_vec(double *vec1, double *vec2, double *result)
{
	#pragma omp for
	for(int i=0;i<n;i++)
	{
		result[i]=vec1[i]-vec2[i];
	}
}

void mul_scalar_vec(double *vec, double scalar)
{
	#pragma omp for
	for(int i=0;i<n;i++)
	{
		vec[i]=vec[i]*scalar;
	}
}
double get_det(double *vec)
{
    //#pragma omp master
	sum=0;
	#pragma omp for reduction(+:sum)
	for(int i=0;i<n;i++)
	{
		sum+=(vec[i]*vec[i]);
	}
	#pragma omp master
	sum=sqrt(sum);
	return sum;
}

void assigment(double *vec1, double *vec2)
{
	#pragma omp for
	for(int i=0;i<n;i++)
	{
		vec1[i]=vec2[i];
	}
}

int main()
{
	omp_set_num_threads(100);
	
	double eps=0.00001;

	double t=0.00001;

	double det=2;

	//std::cin>>n;
	n=5000;

	double *arr;
	arr=(double*)calloc(n*n,sizeof(double));

	double *b;
	b=(double*)calloc(n, sizeof(double));

	double *first_x;
	first_x=(double*)calloc(n, sizeof(double));

	double *tmp_x;
	tmp_x=(double*)calloc(n, sizeof(double));

	double *tmp_x2;
	tmp_x2=(double*)calloc(n, sizeof(double));

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	input_val(arr, b, first_x);

	double det_b=get_det(b);
	
	while(det>eps)
	{
		#pragma omp parallel
	{
		//std::cout<<omp_get_thread_num()<<std::endl;
		
		//#pragma omp single
		mul_mat_vec(arr, first_x, tmp_x);
		//#pragma omp single
		def_vec(tmp_x, b, tmp_x2);
		//#pragma omp single
		mul_scalar_vec(tmp_x2, t);
		//#pragma omp single
		def_vec(first_x, tmp_x2, tmp_x);
		//#pragma omp single
		assigment(first_x, tmp_x);
		//#pragma omp single
		mul_mat_vec(arr, first_x, tmp_x);
		//#pragma omp single
		def_vec(tmp_x, b, tmp_x2);
		//#pragma omp single
		double det1=get_det(tmp_x2);
		det=det1/det_b;
		//std::cout<<det<<std::endl;
	}
    }

	/*for(int i=0;i<n;i++)
	{
		std::cout<<first_x[i]<<" ";
	}
	std::cout<<std::endl;*/

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	printf("Time taken: %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
	free(arr);
	free(b);
	free(tmp_x);
	free(tmp_x2);
	free(first_x);
	return 0;
}
