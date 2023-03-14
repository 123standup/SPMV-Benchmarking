CC=mpicc
FLAG=-O3 -std=c99 -I./include/ -Wno-unused-result -Wno-write-strings
LDFLAG=-O3

OBJS=spmv-mpi.o mmio.o 

.c.o:
	${CC} -o $@ -c ${FLAG} $<

spmv-mpi: ${OBJS}
	${CC}  ${LDFLAG} -o $@ $^

.PHONY:clean
clean: 
	find ./ -name "*.o" -delete
	rm spmv-mpi

