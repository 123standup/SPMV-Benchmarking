CC=gcc
#FLAG=-g -Wall
FLAG=-O3 -std=c99 -I./include/ -Wno-unused-result -Wno-write-strings
LDFLAG=-O3 -fopenmp


OBJS=spmv-omp.o mmio.o 

.c.o:
	${CC} -o $@ -c ${FLAG} $< -fopenmp

spmv-omp: ${OBJS}
	${CC}  ${LDFLAG} -o  $@ $^

.PHONY:clean
clean: 
	find ./ -name "*.o" -delete
	rm spmv-omp

