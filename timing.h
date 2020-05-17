
enum {
    TIME_POTRF = 0,
    TIME_TRSM  = 1,
    TIME_GEMM  = 2,
    TIME_SYRK  = 3,
    TIME_COMM  = 4,
    TIME_CREATE = 5,
    TIME_TOTAL = 6,
    TIME_CNT
};


typedef struct perthread_timing {
    double ts[TIME_CNT];
} perthread_timing_t;

#if USE_NANOS6
static uint64_t ID = 0;
static uint64_t my_next_id() {
  uint64_t ret = __sync_fetch_and_add(&ID, 1);
  return ret;
}
static int get_thread_num(){
  static __thread int thread_num=-1;
  if (thread_num==-1) thread_num=my_next_id();
  return thread_num;
}
static int get_num_threads(){
  return ID;
}
#define THREAD_NUM get_thread_num()
#define NUM_THREADS get_num_threads()
#define MAX_THREADS 50
#else
#define THREAD_NUM omp_get_thread_num()
#define NUM_THREADS omp_get_max_threads()
#define MAX_THREADS omp_get_max_threads()
#endif

#ifdef USE_TIMING

#define INIT_TIMING(nthreads) \
  perthread_timing_t *__timing = calloc(nthreads, sizeof(perthread_timing_t))

#define ACCUMULATE_TIMINGS(nthreads, dst) do{ \
  memset(&(dst), 0, sizeof(dst)); \
  for (int i = 0; i < nthreads; ++i)  \
    for (int j = 0; j < TIME_CNT; ++j) \
      (dst).ts[j] += __timing[i].ts[j];    \
  } while(0)

#define PRINT_TIMINGS() do { \
    perthread_timing_t acc_timings; \
    ACCUMULATE_TIMINGS(NUM_THREADS, acc_timings); \
    printf("[%d] potrf:%f:trsm:%f:gemm:%f:syrk:%f:comm:%f:create:%f:non-calc:%f:total:%f:wall:%f\n", mype, acc_timings.ts[TIME_POTRF], acc_timings.ts[TIME_TRSM],acc_timings.ts[TIME_GEMM],acc_timings.ts[TIME_SYRK],acc_timings.ts[TIME_COMM],acc_timings.ts[TIME_CREATE], acc_timings.ts[TIME_TOTAL]*NUM_THREADS-acc_timings.ts[TIME_POTRF]-acc_timings.ts[TIME_TRSM]-acc_timings.ts[TIME_GEMM]-acc_timings.ts[TIME_SYRK] ,acc_timings.ts[TIME_TOTAL], acc_timings.ts[TIME_TOTAL]*NUM_THREADS); \
    if (mype==0) MPI_Reduce(MPI_IN_PLACE, acc_timings.ts, TIME_CNT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); \
    else         MPI_Reduce(acc_timings.ts, NULL, TIME_CNT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); \
    if (mype==0) printf("[total] potrf:%f:trsm:%f:gemm:%f:syrk:%f:comm:%f:create:%f:non-calc:%f:total:%f:wall:%f\n", acc_timings.ts[TIME_POTRF], acc_timings.ts[TIME_TRSM],acc_timings.ts[TIME_GEMM],acc_timings.ts[TIME_SYRK],acc_timings.ts[TIME_COMM],acc_timings.ts[TIME_CREATE], acc_timings.ts[TIME_TOTAL]*NUM_THREADS-acc_timings.ts[TIME_POTRF]-acc_timings.ts[TIME_TRSM]-acc_timings.ts[TIME_GEMM]-acc_timings.ts[TIME_SYRK] ,acc_timings.ts[TIME_TOTAL], acc_timings.ts[TIME_TOTAL]*NUM_THREADS); \
  } while(0)

#define FREE_TIMING() free(__timing)


#define START_TIMING(timer) double __ts_##timer = timestamp(); int __timer = timer;
#define END_TIMING(timer) __timing[THREAD_NUM].ts[timer] += timestamp() - __ts_##timer

static double timestamp(){
    return MPI_Wtime();
}

#define wait(req) wait_impl(req, &__timing[THREAD_NUM].ts[__timer])
#define waitall(req, nreq) waitall_impl(req, nreq, &__timing[THREAD_NUM].ts[__timer])

static void wait_impl(MPI_Request *comm_req, double *timer)
{
    int comm_comp = 0;

    MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    while (!comm_comp) {
        double yield_time = timestamp();
#ifdef USE_NANOS6
//#pragma oss taskyield
#else
#pragma omp taskyield
#endif
        *timer -= timestamp() - yield_time;
        MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    }
}

static void waitall_impl(MPI_Request *comm_req, int nreq, double *timer)
{
    int comm_comp = 0;

    MPI_Testall(nreq, comm_req, &comm_comp, MPI_STATUS_IGNORE);
    while (!comm_comp) {
        double yield_time = timestamp();
#ifdef USE_NANOS6
//#pragma oss taskyield
#else
#pragma omp taskyield
#endif
        *timer -= timestamp() - yield_time;
        MPI_Testall(nreq, comm_req, &comm_comp, MPI_STATUS_IGNORE);
    }
}


#else 

#define INIT_TIMING(nthreads)
#define FREE_TIMING()
#define START_TIMING(timer)
#define END_TIMING(timer)
#define ACCUMULATE_TIMINGS(nthreads, dst)
#define PRINT_TIMINGS()
#endif


