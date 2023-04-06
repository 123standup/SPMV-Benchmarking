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

#define BLOCK_SIZE 256  

// struct that represents a coord
// used for sorting with qsort
typedef struct {
    int row;
    int col;
    float val;
} coord;

static void usage(char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

__global__
static void child_spmv(int* rows, int* cols, float* vals,
    float* x, float* y_partition, int cnt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cnt)
        atomicAdd(&y_partition[rows[i] - rows[0]], vals[i] * x[cols[i]]);
}

__global__
static void root_spmv(int* rows, int* cols, float* vals,
    float* x, float* y, int cnt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cnt)
        atomicAdd(&y[rows[i]], vals[i] * x[cols[i]]);
}

// one iteration of partial SpMV by a child proc
static void child_iteration(
    int* rows, int* cols, float* vals,
    float* x, float* y_partition, int cnt)
{
    // compute partial product
    int num_blocks = (cnt + BLOCK_SIZE - 1) / BLOCK_SIZE;
    child_spmv<<<num_blocks, BLOCK_SIZE>>>(rows, cols, vals, x, y_partition, cnt);
    cudaDeviceSynchronize();
    
    // send partial product to rank 0
    int num_rows = rows[cnt - 1] - rows[0] + 1;
    MPI_Send(y_partition, num_rows, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
}

// one iteration of SpMV by the root proc
static void root_iteration(
    int num_nonzeros, int* rows, int* cols, float* vals, int* row_indices,
    int row_indices_size, float* x,
    float* y, int size)
{
    // offset into coo vals and # of vals
    int num_rows = row_indices_size / size;
    int cnt = size == 1 ? num_nonzeros : row_indices[num_rows];

    int num_blocks = (cnt + BLOCK_SIZE - 1) / BLOCK_SIZE;
    root_spmv<<<num_blocks, BLOCK_SIZE>>>(rows, cols, vals, x, y, cnt);
    cudaDeviceSynchronize();

    // receive computed y partitions from child procs
    for (int i = 1; i < size; i++) {
        int row_offset = num_rows * i;
        int row_cnt = i < size - 1 ? num_rows : row_indices_size - row_offset;
        int y_offset = rows[row_indices[row_offset]];
        int recv_cnt = rows[row_indices[row_offset + row_cnt - 1]] - y_offset + 1;
        MPI_Recv(y + y_offset, recv_cnt, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// benchmark function
static void benchmark_coo_spmv(
    coo_matrix* coo, int* row_indices,
    int row_indices_size, float* x, float* y)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size = min(size, row_indices_size);

    // device memory
    int *rows;   
    int *cols ;  
    float *vals ;
    cudaMallocManaged(&rows, coo->num_nonzeros * sizeof(int));
    cudaMallocManaged(&cols, coo->num_nonzeros * sizeof(int));
    cudaMallocManaged(&vals, coo->num_nonzeros * sizeof(float));
    cudaMemcpy(rows, coo->rows, coo->num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cols, coo->cols, coo->num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vals, coo->vals, coo->num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);

    // warmup
    timer tw;
    timer_start(&tw);
    root_iteration(coo->num_nonzeros, rows, cols, vals, row_indices, row_indices_size, x, y, size);
    double estimated_time = seconds_elapsed(&tw);
    int num_iterations;
    if (estimated_time == 0)
        num_iterations = MAX_ITER;
    else {
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int) (TIME_LIMIT / estimated_time)) ); 
    }
    printf("\tPerforming %d iterations\n", num_iterations);

    // send # of iterations to all child procs
    for (int i = 1; i < size; i++) {
        MPI_Send(&num_iterations, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    // time several SpMV iterations
    timer t;
    timer_start(&t);
    for (int i = 0; i < num_iterations; i++) {
        root_iteration(coo->num_nonzeros, rows, cols, vals, row_indices, row_indices_size, x, y, size);
    }

    // print out summary stats
    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : (bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs); 
}

// comparison function to sort coords by row
static int coord_cmp(const void* v1, const void* v2)
{
    const coord* c1 = (const coord*) v1;
    const coord* c2 = (const coord*) v2;
    return c1->row - c2->row;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        if (get_arg(argc, argv, "help")) {
            usage(argv);
            MPI_Abort(MPI_COMM_WORLD, EXIT_SUCCESS);
        }

        char* mm_filename = NULL;
        if (argc == 1) {
            printf("Give a MatrixMarket file.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        } else {
            mm_filename = argv[1];
        }

        coo_matrix coo;
        read_coo_matrix(&coo, mm_filename);

    #ifdef TESTING
        // print in COO format
        printf("Writing matrix in COO format to test_COO ...");
        FILE* fp = fopen("test_COO", "w");
        fprintf(fp, "%d\t%d\t%d\n", coo.num_rows, coo.num_cols, coo.num_nonzeros);
        fprintf(fp, "coo.rows:\n");
        for (int i = 0; i < coo.num_nonzeros; i++) {
            fprintf(fp, "%d  ", coo.rows[i]);
        }
        fprintf(fp, "\n\n");
        fprintf(fp, "coo.cols:\n");
        for (int i = 0; i < coo.num_nonzeros; i++) {
            fprintf(fp, "%d  ", coo.cols[i]);
        }
        fprintf(fp, "\n\n");
        fprintf(fp, "coo.vals:\n");
        for (int i = 0; i < coo.num_nonzeros; i++) {
            fprintf(fp, "%f  ", coo.vals[i]);
        }
        fprintf(fp, "\n");
        fclose(fp);
        printf("... done!\n");
    #endif 

        // fill matrix with random values: some matrices have extreme values, 
        // which makes correctness testing difficult, especially in single precision
        srand(13);
        for (int i = 0; i < coo.num_nonzeros; i++) {
            coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
        }

        printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);

        // initialize coords array holding all of the information in coo
        coord* coords = (coord*) malloc(coo.num_nonzeros * sizeof(coord));
        for (int i = 0; i < coo.num_nonzeros; i++) {
            coords[i].row = coo.rows[i];
            coords[i].col = coo.cols[i];
            coords[i].val = coo.vals[i];
        }

        // sort coords with qsort and a custom comparison function to sort by row
        qsort(coords, coo.num_nonzeros, sizeof(coord), coord_cmp);

        // move data in sorted coords back into coo
        for (int i = 0; i < coo.num_nonzeros; i++) {
            coo.rows[i] = coords[i].row;
            coo.cols[i] = coords[i].col;
            coo.vals[i] = coords[i].val;
        }
        
        // table of start indices for each row in the list of values
        int* row_indices = (int*) malloc(coo.num_rows * sizeof(int));
        int prev = -1, row_indices_size = 0;
        for (int i = 0; i < coo.num_nonzeros; i++) {
            if (coo.rows[i] > prev) {
                row_indices[row_indices_size++] = i;
                prev = coo.rows[i];
            }
        }

        // initialize x
        // float* x = (float*) malloc(coo.num_cols * sizeof(float));
        float * x;
        cudaMallocManaged(&x, coo.num_cols * sizeof(float));
        for (int i = 0; i < coo.num_cols; i++) {
            x[i] = rand() / (RAND_MAX + 1.0); 
        }

        // send x to all procs
        // there may be extra procs which are not needed
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        for (int i = 1; i < size; i++) {
            if (i < row_indices_size) {
                MPI_Send(&coo.num_cols, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(x, coo.num_cols, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            } else {
                int terminate = 0;
                MPI_Send(&terminate, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }

        // divide up the rows amongst child procs
        size = min(size, row_indices_size);
        int num_rows = row_indices_size / size;
        for (int i = 1; i < size; i++) {
            // offset into row_indices and # of rows
            int row_offset = num_rows * i;
            int row_cnt = i < size - 1 ? num_rows : row_indices_size - row_offset;

            // offset into coo vals and # of vals
            int offset = row_indices[row_offset];
            int cnt;
            if (row_offset + row_cnt == row_indices_size) {
                cnt = coo.num_nonzeros - row_indices[row_offset];
            } else {
                cnt = row_indices[row_offset + row_cnt] - row_indices[row_offset];
            }

            MPI_Send(&cnt, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(coo.rows + offset, cnt, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(coo.cols + offset, cnt, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(coo.vals + offset, cnt, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }

        // initialize y to all zeros
        // float* y = (float*) calloc(coo.num_rows, sizeof(float));
        float * y;
        cudaMallocManaged(&y, coo.num_rows * sizeof(float));

        // benchmark
        benchmark_coo_spmv(&coo, row_indices, row_indices_size, x, y);

    #ifdef TESTING
        printf("Writing x and y vectors ...");
        fp = fopen("test_x_hybrid", "w");
        for (int i = 0; i < coo.num_cols; i++) {
            fprintf(fp, "%f\n", x[i]);
        }
        fclose(fp);
        fp = fopen("test_y_hybrid", "w");
        for (int i = 0; i < coo.num_rows; i++) {
            fprintf(fp, "%f\n", y[i]);
        }
        fclose(fp);
        printf("... done!\n");
    #endif

        cudaFree(x);
        cudaFree(y);
        free(coords);
        free(row_indices);
        delete_coo_matrix(&coo);

    } else {
        int num_cols;
        MPI_Recv(&num_cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (num_cols == 0) { // extra proc
            MPI_Finalize();
            return EXIT_SUCCESS;
        }

        // receive x
        // float* x = (float*) malloc(num_cols * sizeof(float));
        float * x;
        cudaMallocManaged(&x, num_cols * sizeof(float));
        MPI_Recv(x, num_cols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // receive rows/cols/vals data
        int cnt;
        MPI_Recv(&cnt, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // int* rows = (int*) malloc(cnt * sizeof(int));
        int * rows;
        cudaMallocManaged(&rows, cnt * sizeof(int));
        MPI_Recv(rows, cnt, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // int* cols = (int*) malloc(cnt * sizeof(int));
        int * cols;
        cudaMallocManaged(&cols, cnt * sizeof(int));
        MPI_Recv(cols, cnt, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // float* vals = (float*) malloc(cnt * sizeof(float));
        float * vals;
        cudaMallocManaged(&vals, cnt * sizeof(float));
        MPI_Recv(vals, cnt, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // construct y partition based on rows
        int num_rows = rows[cnt - 1] - rows[0] + 1;
        // float* y_partition = (float*) calloc(num_rows, sizeof(float));
        float * y_partition;
        cudaMallocManaged(&y_partition, num_rows * sizeof(float));
        
        // warmup iteration
        child_iteration(rows, cols, vals, x, y_partition, cnt);

        // receive the number of iterations from the root proc
        int num_iterations;
        MPI_Recv(&num_iterations, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // iterate for num_iterations
        for (int i = 0; i < num_iterations; i++) {
            child_iteration(rows, cols, vals, x, y_partition, cnt);
        }

        cudaFree(x);
        cudaFree(rows);
        cudaFree(cols);
        cudaFree(vals);
        cudaFree(y_partition);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}