
#CC = mpicc
#CFLAGS  += -I${MKLROOT}/include -O3 -I./ -std=gnu99  -fopenmp -g
#LDFLAGS += -std=gnu99 -lrt -lpthread -lm -fopenmp

#LDFLAGS += -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl

#CC = cc
#CFLAGS = -h omp -I${MKLROOT}/include -O3

ifneq ($(TARGET),)
include ../$(TARGET).def
else
TARGET=generic
CFLAGS=-fopenmp
LDFLAGS=-fopenmp
endif

all : ch_$(TARGET)_fine ch_$(TARGET)_fine_extrae ch_$(TARGET)_fine_timing
ch_$(TARGET)_fine: ch_ompss.c ch_common.c Makefile libdetach.so
	$(CC) $(CFLAGS) -o $@ ch_ompss.c -g ch_common.c $(LDFLAGS) -L. -Wl,--rpath,. -ldetach

ch_$(TARGET)_fine_extrae: ch_ompss.c ch_common.c Makefile libdetach.so
	$(CC) $(CFLAGS) -o $@ ch_ompss.c -g ch_common.c $(LDFLAGS) -DUSE_EXTRAE $(EXTRAE_FLAGS) -L. -Wl,--rpath,. -ldetach

ch_$(TARGET)_fine_timing: ch_ompss.c ch_common.c Makefile libdetach.so
	$(CC) $(CFLAGS) -o $@ ch_ompss.c -g ch_common.c $(LDFLAGS) -DUSE_TIMING -L. -Wl,--rpath,. -ldetach

clean:
	rm -f *.o ch_$(TARGET)_fine ch_$(TARGET)_fine_extrae ch_$(TARGET)_fine_timing

cleanall: 
	rm -f *.o ch_*_fine ch_*_fine_extrae ch_*_fine_timing
	
libdetach.so: detach.cpp mpi-detach.h
	$(MPICXX)  -DOMPI_SKIP_MPICXX=1 -shared -fPIC detach.cpp -o libdetach.so -g

