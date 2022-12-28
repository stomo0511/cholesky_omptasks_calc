
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

ifeq ($(TARGET),ompss2_claix)
  SRC=ch_ompss.c ch_common.c
else
  SRC=ch_ompss.c ch_common.c
endif


#all : ch_$(TARGET)_fine ch_$(TARGET)_fine_extrae ch_$(TARGET)_fine_timing
all : ch_$(TARGET)_fine ch_$(TARGET)_fine_timing ch_$(TARGET)_fine_poll

ch_$(TARGET)_fine: $(SRC) ch_common.c Makefile libdetach_$(TARGET).so
	$(CC) $(CFLAGS) -o $@ $(SRC) -g $(LDFLAGS) -L. -Wl,--rpath,$(PWD) -ldetach_$(TARGET)

# ch_$(TARGET)_fine_extrae: $(SRC) ch_common.c Makefile libdetach_$(TARGET).so
#	$(CC) $(CFLAGS) -o $@ $(SRC) -g $(LDFLAGS) -DUSE_EXTRAE $(EXTRAE_FLAGS) -L. -Wl,--rpath,$(PWD) -ldetach_$(TARGET)

ch_$(TARGET)_fine_timing: $(SRC) ch_common.c Makefile libdetach_$(TARGET).so
	$(CC) $(CFLAGS) -o $@ $(SRC) -g $(LDFLAGS) -DUSE_TIMING -L. -Wl,--rpath,$(PWD) -ldetach_$(TARGET)

ch_$(TARGET)_fine_poll: $(SRC) ch_common.c Makefile libdetach_$(TARGET).so
	$(CC) $(CFLAGS) -o $@ $(SRC) -g $(LDFLAGS) -DUSE_POLLING -DUSE_TIMING -L. -Wl,--rpath,$(PWD) -ldetach_$(TARGET)

clean:
	rm -f *.o ch_$(TARGET)_fine ch_$(TARGET)_fine_extrae ch_$(TARGET)_fine_timing

cleanall: 
	rm -f *.o ch_*_fine ch_*_fine_extrae ch_*_fine_timing
	
libdetach_$(TARGET).so: detach.cpp mpi-detach.h Makefile
	$(MPICXX)  -DOMPI_SKIP_MPICXX=1 -std=c++14 -shared -fPIC detach.cpp -o libdetach_$(TARGET).so -g

