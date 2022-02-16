
#CC = mpicc
#CFLAGS  += -I${MKLROOT}/include -O3 -I./ -std=gnu99  -fopenmp -g
#LDFLAGS += -std=gnu99 -lrt -lpthread -lm -fopenmp

#LDFLAGS += -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl


#CC = cc -craype-verbose
#CFLAGS = -fopenmp -I${MKLROOT}/include -O3

ifneq ($(TARGET),)
include ../$(TARGET).def
else
TARGET=generic
CFLAGS=-fopenmp
LDFLAGS=-fopenmp
endif

#CFLAGS+= -DFAKEMPI -L/home/pj416018/sw/UTIL/clang -Wl,--rpath,/home/pj416018/sw/UTIL/clang -lompt_fakempicriticalpath
#CFLAGS+= 

all : ch_$(TARGET)_single ch_$(TARGET)_single_noyield ch_$(TARGET)_single_timing Makefile
ch_$(TARGET)_single: cholesky.c ch_common.c Makefile
	$(CC) $(CFLAGS) -o $@ cholesky.c ch_common.c $(LDFLAGS)

ch_$(TARGET)_single_noyield: cholesky.c ch_common.c Makefile
	$(CC) $(CFLAGS) -DDISABLE_TASKYIELD -o $@ cholesky.c ch_common.c $(LDFLAGS)

ch_$(TARGET)_single_timing: cholesky.c ch_common.c Makefile
	$(CC) $(CFLAGS) -DDISABLE_TASKYIELD -o $@ cholesky.c ch_common.c $(LDFLAGS) -DUSE_TIMING

clean:
	rm -f *.o ch_$(TARGET)_single ch_$(TARGET)_single_noyield ch_$(TARGET)_single_timing

cleanall: 
	rm -f *.o ch_*_single ch_*_single_noyield ch_*_single_timing
