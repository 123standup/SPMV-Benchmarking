NVCC=nvcc
CC=g++
CFLAGS=-O3 -I./include/ -Wno-unused-result -Wno-write-strings
NVCCFLAGS=-O3 -I./include/
NVCCMPIFLAGS=-O3 -I./include/ -I/opt/ohpc/pub/mpi/mvapich2-gnu/2.2/include -L/opt/ohpc/pub/mpi/mvapich2-gnu/2.2/lib -lmpi -DMPICH_IGNORE_CXX_SEEK

all: spmv-cuda spmv-mpi-cuda

spmv-mpi-cuda.o: spmv-mpi-cuda.cu
	$(NVCC) $(NVCCMPIFLAGS) -c -o $@ $<

spmv-cuda.o: spmv-cuda.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

mmio.o: mmio.c
	$(CC) $(CFLAGS) -c -o $@ $<

spmv-cuda: spmv-cuda.o mmio.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^

spmv-mpi-cuda: spmv-mpi-cuda.o mmio.o
	$(NVCC) $(NVCCMPIFLAGS) -o $@ $^

clean:
	rm -f spmv-cuda.o mmio.o spmv-mpi-cuda.o spmv-cuda spmv-mpi-cuda
	
