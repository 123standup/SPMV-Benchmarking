#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
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

void partition_by_row(coo_matrix * coo, coo_matrix * par, int num_procs, int num_threads)
{
    int total = num_procs * num_threads;
    int *arr[total];
    int *cnt = (int*) malloc(total * sizeof(int));
    // Initialize
    for (int i = 0; i < total; i ++)
    {
        arr[i] = (int*) malloc(coo->num_nonzeros * sizeof(int));
        cnt[i] = 0;
    }
    // Partition row to each process by row index
    int num_partition = coo->num_rows / total + 1;
    for (int i = 0; i < coo->num_nonzeros; i ++)
    {
        int index = (coo->rows[i]) / num_partition;
        arr[index][cnt[index]] = i;
        cnt[index] ++;
    }
    // Send each partition to each process
    for (int i = 0; i < total; i ++)
    {
        int send_to = i / num_threads;
        int par_idx = i % num_threads;
        int num_nonzeros = cnt[i];
        if (send_to != 0)
            MPI_Send(&num_nonzeros, 1, MPI_INT, send_to, 0, MPI_COMM_WORLD);
        int* rows =   (int*) malloc(num_nonzeros * sizeof(int));
        int* cols =   (int*) malloc(num_nonzeros * sizeof(int));
        float* vals = (float*) malloc(num_nonzeros * sizeof(float));
        for (int j = 0; j < num_nonzeros; j ++)
        {
            rows[j] = coo->rows[arr[i][j]];
            cols[j] = coo->cols[arr[i][j]];
            vals[j] = coo->vals[arr[i][j]];
        }
        if (send_to != 0 && num_nonzeros > 0)
        {
            MPI_Send(rows, num_nonzeros, MPI_INT, send_to, 0, MPI_COMM_WORLD);
            MPI_Send(cols, num_nonzeros, MPI_INT, send_to, 0, MPI_COMM_WORLD);
            MPI_Send(vals, num_nonzeros, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);
        }
        else if(send_to == 0 && num_nonzeros > 0)
        {
            par[par_idx].num_nonzeros = num_nonzeros;
            par[par_idx].rows = (int*)   malloc(par[par_idx].num_nonzeros * sizeof(int));
            par[par_idx].cols = (int*)   malloc(par[par_idx].num_nonzeros * sizeof(int));
            par[par_idx].vals = (float*) malloc(par[par_idx].num_nonzeros * sizeof(float));
            par[par_idx].rows = rows;
            par[par_idx].cols = cols;
            par[par_idx].vals = vals;
        }
    }
    for (int i = 0; i < total; i ++) free(arr[i]);
    free(cnt);
}

int main(int argc, char** argv)
{
    int ierr, num_procs, rank;
    coo_matrix coo;
    FILE *fp;

    // parallel starts
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    float *x, *y;
    timer excute_time;
    // get numbers of thread
    int num_threads;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    coo_matrix *par = (coo_matrix*) malloc(num_threads * sizeof(coo_matrix));

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
        y = (float*) malloc(coo.num_rows * num_threads * sizeof(float));
        for(int i = 0; i < coo.num_rows * num_threads; i ++)
            y[i] = 0;
        for (int i = 0; i < coo.num_cols; i ++)
            x[i] = rand() / (RAND_MAX + 1.0);

        for (int i = rank + 1; i < num_procs; i ++)
        {
            for (int j = 0; j < num_threads; j ++)
            {
                MPI_Send(&coo.num_rows, 1, MPI_INT, i, j, MPI_COMM_WORLD);
                MPI_Send(&coo.num_cols, 1, MPI_INT, i, j, MPI_COMM_WORLD);
            }
            MPI_Send(x, coo.num_cols, MPI_FLOAT, i, i, MPI_COMM_WORLD);
        }
        for (int i = 0; i < num_threads; i ++)
        {
            par[i].num_rows = coo.num_rows;
            par[i].num_cols = coo.num_cols; 
        }
    }
    else
    {
        for (int i = 0; i < num_threads; i ++)
        {
            MPI_Recv(&par[i].num_rows, 1, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&par[i].num_cols, 1, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        x = (float*) malloc(par[0].num_cols * sizeof(float));
        y = (float*) malloc(par[0].num_rows * num_threads * sizeof(float));
        MPI_Recv(x, par[0].num_cols, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int i = 0; i < par[0].num_rows * num_threads; i++)
            y[i] = 0;
    }
    if (rank == 0)
    {
        partition_by_row(&coo, par, num_procs, num_threads);
    }
    else
    {
        for (int i = 0; i < num_threads; i ++)
        {
            MPI_Recv(&par[i].num_nonzeros, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (par[i].num_nonzeros > 0)
            {
                par[i].rows = (int*)   malloc(par[i].num_nonzeros * sizeof(int));
                par[i].cols = (int*)   malloc(par[i].num_nonzeros * sizeof(int));
                par[i].vals = (float*) malloc(par[i].num_nonzeros * sizeof(float));
                MPI_Recv(par[i].rows, par[i].num_nonzeros, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(par[i].cols, par[i].num_nonzeros, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(par[i].vals, par[i].num_nonzeros, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    if (rank == 0)
        timer_start(&excute_time);
    float *sum = (float*) malloc(par[0].num_rows * sizeof(float));

    for (int k = 0; k <= MIN_ITER; k ++)
    {
        /* Benchmarking */
        // pass matrix to threads on each process
        #pragma omp parallel
        {
            double coo_gflops;
            int thd_id = omp_get_thread_num(); 
            if (par[thd_id].num_nonzeros > 0)
                benchmark_coo_spmv(&par[thd_id], x, y + thd_id * par[thd_id].num_cols);
        }
        // sum up thread result
        for (int j = 0; j < num_threads; j ++)
        {
            for (int i = 0; i < par[0].num_cols; i ++)
            {
                // y[i] += y[j * par[0].num_cols + i];
                sum[i] += y[j * par[0].num_cols + i];
                y[j * par[0].num_cols + i] = 0.0;
            }
        }
        // child process send each result to master process
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
        }
        else
        {
            MPI_Send(sum, par[0].num_rows, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
            for(int i = 0; i < par[0].num_rows; i ++) sum[i] = 0.0;
        }
    }

    if (rank == 0)
        printf("Total time elapsed: %8.4f ms\n", milliseconds_elapsed(&excute_time));

    // free memory
    if (rank == 0)
    {
        free(fp);
        delete_coo_matrix(&coo);
    }
    free(x);
    free(y);
    free(par);
    free(sum);
    MPI_Finalize();

    return 0;
}

