#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

void benchmark_coo_spmv(coo_matrix * coo, float* x, float* y)
{
    int num_nonzeros = coo->num_nonzeros;

    for (int i = 0; i < num_nonzeros; i++){   
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }
}

void partition_by_row(coo_matrix * coo, coo_matrix * par, int num_procs)
{
    int *arr[num_procs];
    int *cnt = (int*) malloc(num_procs * sizeof(int));
    // Initialize
    for (int i = 0; i < num_procs; i ++)
    {
        arr[i] = (int*) malloc(coo->num_nonzeros * sizeof(int));
        cnt[i] = 0;
    }
    // Partition row to each process by row index
    int num_partition = coo->num_rows / num_procs + 1;
    for (int i = 0; i < coo->num_nonzeros; i ++)
    {
        int index = (coo->rows[i]) / num_partition;
        arr[index][cnt[index]] = i;
        cnt[index] ++;
    }
    // Send each partition to each process
    for (int i = 0; i < num_procs; i ++)
    {
        int num_nonzeros = cnt[i];
        if (i != 0)
            MPI_Send(&num_nonzeros, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        int* rows =   (int*) malloc(num_nonzeros * sizeof(int));
        int* cols =   (int*) malloc(num_nonzeros * sizeof(int));
        float* vals = (float*) malloc(num_nonzeros * sizeof(float));
        for (int j = 0; j < num_nonzeros; j ++)
        {
            rows[j] = coo->rows[arr[i][j]];
            cols[j] = coo->cols[arr[i][j]];
            vals[j] = coo->vals[arr[i][j]];
        }
        if (i != 0 && num_nonzeros > 0)
        {
            MPI_Send(rows, num_nonzeros, MPI_INT, i, i + 2, MPI_COMM_WORLD);
            MPI_Send(cols, num_nonzeros, MPI_INT, i, i + 3, MPI_COMM_WORLD);
            MPI_Send(vals, num_nonzeros, MPI_FLOAT, i, i + 4, MPI_COMM_WORLD);
        }
        else if(i == 0)
        {
            par->num_nonzeros = num_nonzeros;
            par->num_rows = coo->num_rows;
            par->num_cols = coo->num_cols;
            par->rows = (int*) malloc(par->num_nonzeros * sizeof(int));
            par->cols = (int*) malloc(par->num_nonzeros * sizeof(int));
            par->vals = (float*) malloc(par->num_nonzeros * sizeof(float));
            par->rows = rows;
            par->cols = cols;
            par->vals = vals;
        }
    }
    free(arr);
    free(cnt);
}

int main(int argc, char** argv)
{
    int ierr, num_procs, rank;
    float *x, *y;
    coo_matrix coo, par;
    FILE *fp;
    timer excute_time;
    // parallel starts
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (rank == 0)
    {
        if (get_arg(argc, argv, "help") != NULL){
            usage(argc, argv);
            return 0;
        }

        char * mm_filename = NULL;
        if (argc == 1) {
            printf("Give a MatrixMarket file.\n");
            return -1;
        } else 
            mm_filename = argv[1];

        read_coo_matrix(&coo, mm_filename);
        printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
        fflush(stdout);
    }
    
    // initialize arrays
    if (rank == 0)
    {
        x = (float*) malloc(coo.num_cols * sizeof(float));
        y = (float*) malloc(coo.num_rows * sizeof(float));
        for (int i = 0; i < coo.num_cols; i ++)
            x[i] = rand() / (RAND_MAX + 1.0);
        for(int i = 0; i < coo.num_rows; i++)
            y[i] = 0;
        for (int i = rank + 1; i < num_procs; i ++)
        {
            MPI_Send(&coo.num_rows, 1, MPI_INT, i, i + 1, MPI_COMM_WORLD);
            MPI_Send(&coo.num_cols, 1, MPI_INT, i, i + 2, MPI_COMM_WORLD);
            MPI_Send(x, coo.num_cols, MPI_FLOAT, i, i + 3, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&par.num_rows, 1, MPI_INT, 0, rank + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&par.num_cols, 1, MPI_INT, 0, rank + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        x = (float*) malloc(par.num_cols * sizeof(float));
        y = (float*) malloc(par.num_rows * sizeof(float));
        MPI_Recv(x, par.num_cols, MPI_FLOAT, 0, rank + 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int i = 0; i < par.num_rows; i++)
            y[i] = 0;
    }
    if (rank == 0)
    {
        partition_by_row(&coo, &par, num_procs);
    }
    else
    {
        MPI_Recv(&par.num_nonzeros, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (par.num_nonzeros > 0)
        {
            par.rows = (int*) malloc(par.num_nonzeros * sizeof(int));
            par.cols = (int*) malloc(par.num_nonzeros * sizeof(int));
            par.vals = (float*) malloc(par.num_nonzeros * sizeof(float));
            MPI_Recv(par.rows, par.num_nonzeros, MPI_INT, 0, rank + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(par.cols, par.num_nonzeros, MPI_INT, 0, rank + 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(par.vals, par.num_nonzeros, MPI_FLOAT, 0, rank + 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    /* Benchmarking */
    if (rank == 0)
        timer_start(&excute_time);
    float *sum = (float*) malloc(par.num_rows * sizeof(float));
    
    for (int k = 0; k <= MIN_ITER; k ++)
    {
        double coo_gflops;
        if (par.num_nonzeros > 0)
            benchmark_coo_spmv(&par, x, y);
        
        if (rank == 0)
        {
            for (int i = 1; i < num_procs; i ++)
            {
                float *temp = (float*) malloc(coo.num_rows * sizeof(float));
                MPI_Recv(temp, coo.num_rows, MPI_FLOAT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < coo.num_rows; j ++)
                    sum[j] += temp[j];
                free(temp);
            }
            for (int j = 0; j < coo.num_rows; j ++)
            {
                sum[j] += y[j];
                y[j] = 0;
            }
        }
        else
        {
            MPI_Send(y, par.num_rows, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
            for (int j = 0; j < par.num_rows; j ++) y[j] = 0;
        }
    }
    
    if (rank == 0)
        printf("Total time elapsed: %8.4f ms\n", milliseconds_elapsed(&excute_time));

    free(x);
    free(y);
    free(sum);
    delete_coo_matrix(&coo);
    delete_coo_matrix(&par);
    MPI_Finalize();

    return 0;
}

