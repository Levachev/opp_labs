#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
//2048
#define N1 512
#define N2 512
#define N3 512

void init_A(double *A){
	for(int i=0;i<N1;i++){
		for(int j=0;j<N2;j++){
			A[i*N2+j]=i+j;
		}
	}
}

void init_B(double *B){
	for(int i=0;i<N2;i++){
		for(int j=0;j<N3;j++){
			B[i*N3+j]=i*j;
		}
	}
}	

void mul_mat(double *local_A, double *local_B, double *local_C, int local_ny, int local_nx)
{
	for(int i=0;i<local_ny;i++)
	{
		for(int j=0;j<local_nx;j++)
		{
			local_C[i*local_nx+j]=0;
			for(int k=0;k<N2;k++)
			{
				local_C[i*local_nx+j]+=local_A[i*N2+k]*local_B[k*local_nx+j];
			}
		}
	}
}


int main(int argc, char *argv[])
{

	MPI_Init(&argc, &argv);

	int size;

	int dims[2]={0,0},periods[2]={0,0},coords[2],reorder=1;
	int rank,sizey,sizex,ranky,rankx;

	MPI_Comm comm2d;

	MPI_Comm row_comm;

	MPI_Comm col_comm;

	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Dims_create(size,2,dims);

	sizey = dims[0]; sizex = dims[1];

	MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,reorder,&comm2d);
	
	MPI_Comm_rank(comm2d,&rank);


	MPI_Cart_get(comm2d,2,dims,periods,coords);
	ranky=coords[0]; rankx=coords[1];

	MPI_Comm_split(comm2d, rankx, 0, &col_comm);
	MPI_Comm_split(comm2d, ranky, 0, &row_comm);
	int csize, rsize;

	MPI_Comm_size(col_comm,&csize);
	MPI_Comm_size(row_comm,&rsize);

	/*printf("csize - %d\n", csize);
	printf("rsize - %d\n", rsize);*/


	int local_ny=N1/sizey;
	int local_nx=N3/sizex;

	/*printf("sizey - %d\n", sizey);
	printf("sizex - %d\n", sizex);
	printf("local_ny - %d\n", local_ny);
	printf("local_nx - %d\n", local_nx);*/

	double t1, t2;
     t1 = MPI_Wtime();

	double *local_A;
	local_A=(double*)calloc(local_ny*N2, sizeof(double));

	double *local_B;
	local_B=(double*)calloc(N2*local_nx, sizeof(double));

	double *local_C;
	local_C=(double*)calloc(local_ny*local_nx, sizeof(double));



	double *A;
	A=(double*)calloc(N1*N2, sizeof(double));

	double *B;
	B=(double*)calloc(N2*N3, sizeof(double));

	double *C;
	C=(double*)calloc(N1*N3, sizeof(double));

	double *C_row;
	C_row=(double*)calloc(local_ny*N3, sizeof(double));
	
	init_A(A);
	
	init_B(B);

	if(rank==60)
	{
		for(int i=0;i<N1;i++)
		{
			for(int j=0;j<N2;j++)
			{
				printf("A-%lf ",A[i*N2+j]);
			}
			printf("\n");
		}

		printf("\n");
		printf("\n");
		printf("\n");


		for(int i=0;i<N2;i++)
		{
			for(int j=0;j<N3;j++)
			{
				printf("B-%lf ",B[i*N3+j]);
			}
			printf("\n");
		}
	}
	MPI_Datatype col_type, lcol_type, new_t;
	MPI_Type_vector(N2,local_nx,N3,MPI_DOUBLE,&col_type);
	MPI_Type_commit(&col_type);

	MPI_Type_vector(N2,local_nx,local_nx,MPI_DOUBLE,&lcol_type);
	MPI_Type_commit(&lcol_type);

	MPI_Type_create_resized(col_type, 0, 8*local_nx, &new_t);
    MPI_Type_commit(&new_t);

/*     MPI_Aint lb, ub, extent;

     MPI_Type_get_extent(col_type, &lb, &extent);

     printf("%d %d\n", lb, extent);

     MPI_Type_get_extent(new_t, &lb, &extent);

     printf("2-%d %d\n", lb, extent);

     MPI_Finalize();

	return 0;*/


	MPI_Scatter(B, 1, new_t, local_B, 1, lcol_type, 0, row_comm);
	MPI_Scatter(A, local_ny*N2, MPI_DOUBLE, local_A, local_ny*N2, MPI_DOUBLE, 0, col_comm);

	/*if(rank==0)
	{
		for(int i=0;i<N2;i++){
			for(int j=0;j<local_nx;j++){
				printf("%lf ", local_B[i*local_nx+j]);
			}
			printf("\n");
		}
	}*/

	mul_mat(local_A, local_B, local_C,local_ny, local_nx);

	MPI_Datatype mat, r_mat, big_row, r_big_row;

	MPI_Type_vector(local_ny,local_nx,local_nx,MPI_DOUBLE,&mat);
	MPI_Type_commit(&mat);

	MPI_Type_create_resized(mat, 0, 8*local_nx, &r_mat);
    MPI_Type_commit(&r_mat);

    MPI_Type_vector(local_ny,local_nx,N3,MPI_DOUBLE,&big_row);
	MPI_Type_commit(&big_row);

	MPI_Type_create_resized(big_row, 0, 8*local_nx, &r_big_row);
    MPI_Type_commit(&r_big_row);

	MPI_Gather(local_C, 1, r_mat, C_row, 1, r_big_row, 0, row_comm);

	/*int rank_row;
	MPI_Comm_rank(row_comm,&rank);
	if(rank_row==0)
	{
		for(int i=0;i<local_ny;i++)
		{
			for(int j=0;j<N3;j++)
			{
				printf("%lf ",C_row[i*N3+j]);
			}
			printf("\n");
		}
		printf("\n");
		printf("\n");
	}*/
	

    if(rankx==0)
    {
    	MPI_Gather(C_row, local_ny*N3, MPI_DOUBLE, C, local_ny*N3, MPI_DOUBLE, 0, col_comm);
    }


	/*for(int i=0;i<local_ny;i++)
	{
		for(int j=0;j<local_nx;j++)
		{
			int newy=ranky*local_ny+i;
			int newX=rankx*local_nx+j;
			C[newy*N3+newX]=local_C[i*local_nx+j];
		}
	}

	MPI_Reduce(C, C1, N1*N3, MPI_DOUBLE, MPI_SUM, 0, comm2d);*/

	/*if(rank==0)
	{
		for(int i=0;i<N1;i++)
		{
			for(int j=0;j<N3;j++)
			{
				printf("%lf ",C[i*N3+j]);
			}
			printf("\n");
		}

	}*/
    t2 = MPI_Wtime();
    printf( "Elapsed time is %f\n", t2 - t1 );

	MPI_Finalize();

	return 0;
}
