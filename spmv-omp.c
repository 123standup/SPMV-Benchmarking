
#include <stdio.h>
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

#define NUM_THREADS 8

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

void partition_by_row(coo_matrix * coo, coo_matrix * par, int num_threads)
{
    int *arr[num_threads];
    int *cnt = (int*) malloc(num_threads * sizeof(int));
    // Initialize
    for (int i = 0; i < num_threads; i ++)
    {
        arr[i] = (int*) malloc(coo->num_nonzeros * sizeof(int));
        cnt[i] = 0;
    }
    // Partition row to each process by row index
    int num_partition = coo->num_rows / num_threads + 1;
    for (int i = 0; i < coo->num_nonzeros; i ++)
    {
        int index = (coo->rows[i]) / num_partition;
        arr[index][cnt[index]] = i;
        cnt[index] ++;
    }
    // Store partiton data into par
    for (int i = 0; i < num_threads; i ++)
    {
        int num_nonzeros = cnt[i];
        par[i].num_nonzeros = cnt[i];
        par[i].num_rows = coo->num_rows;
        par[i].num_cols = coo->num_cols;
        par[i].rows =   (int*) malloc(num_nonzeros * sizeof(int));
        par[i].cols =   (int*) malloc(num_nonzeros * sizeof(int));
        par[i].vals = (float*) malloc(num_nonzeros * sizeof(float));
        for (int j = 0; j < num_nonzeros; j ++)
        {
            par[i].rows[j] = coo->rows[arr[i][j]];
            par[i].cols[j] = coo->cols[arr[i][j]];
            par[i].vals[j] = coo->vals[arr[i][j]];
        }
    }
    free(cnt);
    for (int i = 0; i < num_threads; i ++)
        free(arr[i]);
}


int main(int argc, char** argv)
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

    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);
    timer excute_time;

    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

    //initialize host arrays
    float * x = (float*)malloc(coo.num_cols * sizeof(float));
    for(int i = 0; i < coo.num_cols; i++) {
        x[i] = rand() / (RAND_MAX + 1.0); 
    }

    omp_set_num_threads(NUM_THREADS);
    int num_threads;
    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            num_threads = omp_get_num_threads();
    }
    printf("num_threads: %d\n", num_threads);

    coo_matrix *par = (coo_matrix *) malloc(num_threads * sizeof(coo_matrix)); 
    partition_by_row(&coo, par, num_threads);

    timer_start(&excute_time);
    float *sum = (float*) malloc(par[0].num_rows * sizeof(float));
    for (int k = 0; k <= MIN_ITER; k ++)
    {
        /* Benchmarking */
        float *y[num_threads];
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            y[id] = (float*) malloc(coo.num_rows * sizeof(float));
            for (int i = 0; i < coo.num_rows; i ++) y[id][i] = 0;
            if (par[id].num_nonzeros > 0)
                benchmark_coo_spmv(&par[id], x, y[id]);
        }
    
        // add all result to y
        for (int i = 0; i < num_threads; i ++)
        {
            for (int j = 0; j < coo.num_rows; j ++)
                sum[j] += y[i][j];
            free(y[i]);
        }
    }
    printf("Total time elapsed: %8.4f ms\n", milliseconds_elapsed(&excute_time));
    delete_coo_matrix(&coo);
    free(x);

    return 0;
}

