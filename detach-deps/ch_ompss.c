
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include "ch_common.h"
#include "mpi-detach.h"
#include "../extrae.h"
#include "../timing.h"

//#ifdef _OMPSS
//#warning "Compiling for OMPSS"
//#endif

#ifdef USE_NANOS6
#include "nanos6.h"
int omp_fulfill_event(void* arg){
  nanos6_decrease_task_event_counter(arg, 1);
}
#endif

void Detach_callback(void * data, MPI_Request * req){
    omp_fulfill_event(data);
}
void Detach_all_callback(void * data, int count, MPI_Request req[]){
    omp_fulfill_event(data);
}


static int comm_round_sentinel; // <-- used to limit parallel communication tasks

void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt], int *block_rank)
{
	REGISTER_EXTRAE();

#ifndef USE_NANOS6
#pragma omp parallel
#pragma omp single
#endif
{
    INIT_TIMING(NUM_THREADS);
    char *send_flags = malloc(sizeof(char) * np);
    char recv_flag = 0;
    int num_send_tasks = 0;
    int num_recv_tasks = 0;
    int max_send_tasks = 0;
    int max_recv_tasks = 0;
    int num_comp_tasks = 0;
    reset_send_flags(send_flags);

    START_TIMING(TIME_TOTAL);
    {
    START_TIMING(TIME_CREATE);
    for (int k = 0; k < nt; k++) {
        int send_tasks = 0, recv_tasks = 0;
        // sentinel task to limit communication task parallelism
#ifdef HAVE_COMM_SENTINEL
#ifdef USE_NANOS6
#pragma oss task depend(out: comm_round_sentinel)
#else
#pragma omp task depend(out: comm_round_sentinel)
#endif
        { if (comm_round_sentinel < 0) comm_round_sentinel = 0; }
#endif // HAVE_COMM_SENTINEL
        if (block_rank[k*nt+k] == mype) {
            num_comp_tasks++;
#ifdef USE_NANOS6
#pragma oss task depend(out: A[k][k]) firstprivate(k)
#else
#pragma omp task depend(out: A[k][k]) firstprivate(k)
#endif
{
			EXTRAE_ENTER(EVENT_POTRF);
			START_TIMING(TIME_POTRF);
            omp_potrf(A[k][k], ts, ts);
			END_TIMING(TIME_POTRF);
			EXTRAE_EXIT(EVENT_POTRF);
}
        }

        if (block_rank[k*nt+k] == mype && np != 1) {
#ifdef USE_NANOS6
#pragma oss task depend(in: A[k][k]) firstprivate(k) depend(in: comm_round_sentinel) untied
{
            void * event_handle = nanos6_get_current_event_counter();
            nanos6_increase_current_task_event_counter(event_handle, 1);
#else
                       omp_event_handle_t event_handle;
#pragma omp task depend(in: A[k][k]) firstprivate(k) depend(in: comm_round_sentinel) detach(event_handle)
{
#endif
            START_TIMING(TIME_COMM);
            MPI_Request reqs[np];
            int nreqs = 0;
            for (int dst = 0; dst < np; dst++) {
                int send_flag = 0;
                for (int kk = k+1; kk < nt; kk++) {
                   if (dst == block_rank[k*nt+kk]) { send_flag = 1; break; }
                }
                if (send_flag && dst != mype) {
                    MPI_Request send_req;
                    MPI_Isend(A[k][k], ts*ts, MPI_DOUBLE, dst, k*nt+k, MPI_COMM_WORLD, &send_req);
                    reqs[nreqs++] = send_req;
                }
            }
            MPIX_Detach_all(nreqs, reqs, Detach_all_callback, (void*)event_handle);

			END_TIMING(TIME_COMM);
}
        }

        if (block_rank[k*nt+k] != mype) {
            for (int i = k + 1; i < nt; i++) {
                if (block_rank[k*nt+i] == mype) recv_flag = 1;
            }
            if (recv_flag) {
#ifdef USE_NANOS6
#pragma oss task depend(out: B) firstprivate(k) depend(in: comm_round_sentinel) untied 
{
            void * event_handle = nanos6_get_current_event_counter();
            nanos6_increase_current_task_event_counter(event_handle, 1);
#else
                       omp_event_handle_t event_handle;
#pragma omp task depend(out: B) firstprivate(k) depend(in: comm_round_sentinel) detach(event_handle)// untied
{
#endif
            START_TIMING(TIME_COMM);
                MPI_Request recv_req;
                MPI_Irecv(B, ts*ts, MPI_DOUBLE, block_rank[k*nt+k], k*nt+k, MPI_COMM_WORLD, &recv_req);
                MPIX_Detach(&recv_req, Detach_callback, (void*)event_handle);
			END_TIMING(TIME_COMM);
}
                recv_flag = 0;
            }
        }

#ifdef HAVE_INTERMEDIATE_COMM_SENTINEL
        // sentinel task to limit communication task parallelism
#ifdef USE_NANOS6
#pragma oss task depend(out: comm_round_sentinel)
#else
#pragma omp task depend(out: comm_round_sentinel)
#endif
        { if (comm_round_sentinel < 0) comm_round_sentinel = 0; }
#endif

        for (int i = k + 1; i < nt; i++) {
            if (block_rank[k*nt+i] == mype) {
                num_comp_tasks++;
                if (block_rank[k*nt+k] == mype) {
#ifdef USE_NANOS6
#pragma oss task depend(in: A[k][k]) depend(out: A[k][i]) firstprivate(k, i)
#else
#pragma omp task depend(in: A[k][k]) depend(out: A[k][i]) firstprivate(k, i)
#endif
{
					EXTRAE_ENTER(EVENT_TRSM);
			        START_TIMING(TIME_TRSM);
                    omp_trsm(A[k][k], A[k][i], ts, ts);
			        END_TIMING(TIME_TRSM);
					EXTRAE_EXIT(EVENT_TRSM);
}
                } else {
#ifdef USE_NANOS6
#pragma oss task depend(in: B) depend(out: A[k][i]) firstprivate(k, i)
#else
#pragma omp task depend(in: B) depend(out: A[k][i]) firstprivate(k, i)
#endif
{
                    EXTRAE_ENTER(EVENT_TRSM);
			        START_TIMING(TIME_TRSM);
                    omp_trsm(B, A[k][i], ts, ts);
                    END_TIMING(TIME_TRSM);
                    EXTRAE_EXIT(EVENT_TRSM);
}
                }
            }

            if (block_rank[k*nt+i] == mype && np != 1) {
                for (int ii = k + 1; ii < i; ii++) {
                    if (!send_flags[block_rank[ii*nt+i]]) send_flags[block_rank[ii*nt+i]] = 1;
                }
                for (int ii = i + 1; ii < nt; ii++) {
                    if (!send_flags[block_rank[i*nt+ii]]) send_flags[block_rank[i*nt+ii]] = 1;
                }
                if (!send_flags[block_rank[i*nt+i]]) send_flags[block_rank[i*nt+i]] = 1;
                for (int dst = 0; dst < np; dst++) {
                    if (send_flags[dst] && dst != mype) {
                       send_tasks++;
                       num_send_tasks++;
#ifdef USE_NANOS6
#pragma oss task depend(in: A[k][i]) firstprivate(k, i, dst) depend(in: comm_round_sentinel) untied 
{
            void * event_handle = nanos6_get_current_event_counter();
            nanos6_increase_current_task_event_counter(event_handle, 1);
#else
                       omp_event_handle_t event_handle;
#pragma omp task depend(in: A[k][i]) firstprivate(k, i, dst) depend(in: comm_round_sentinel) detach(event_handle) //untied 
{
#endif
            START_TIMING(TIME_COMM);
                        MPI_Request send_req;
                        MPI_Isend(A[k][i], ts*ts, MPI_DOUBLE, dst, k*nt+i, MPI_COMM_WORLD, &send_req);
                        MPIX_Detach(&send_req, Detach_callback, (void*)event_handle);
			END_TIMING(TIME_COMM);
}
                    }
                }
                reset_send_flags(send_flags);
            }
            if (block_rank[k*nt+i] != mype) {
                for (int ii = k + 1; ii < i; ii++) {
                    if (block_rank[ii*nt+i] == mype) recv_flag = 1;
                }
                for (int ii = i + 1; ii < nt; ii++) {
                    if (block_rank[i*nt+ii] == mype) recv_flag = 1;
                }
                if (block_rank[i*nt+i] == mype) recv_flag = 1;
                if (recv_flag) {
                    recv_tasks++;
                    num_recv_tasks++;
#ifdef USE_NANOS6
#pragma oss task depend(out: C[i]) firstprivate(k, i) depend(in: comm_round_sentinel) untied
{
            void * event_handle = nanos6_get_current_event_counter();
            nanos6_increase_current_task_event_counter(event_handle, 1);
#else
                       omp_event_handle_t event_handle;
#pragma omp task depend(out: C[i]) firstprivate(k, i) depend(in: comm_round_sentinel) detach(event_handle)
{
#endif
            START_TIMING(TIME_COMM);
                    MPI_Request recv_req;
                    MPI_Irecv(C[i], ts*ts, MPI_DOUBLE, block_rank[k*nt+i], k*nt+i, MPI_COMM_WORLD, &recv_req);
                    MPIX_Detach(&recv_req, Detach_callback, (void*)event_handle);
			END_TIMING(TIME_COMM);
}
                    recv_flag = 0;
                }
            }
        }

        if ((max_send_tasks + max_recv_tasks) < (send_tasks + recv_tasks)) {
          max_send_tasks = send_tasks;
          max_recv_tasks = recv_tasks;
        }

        for (int i = k + 1; i < nt; i++) {

            for (int j = k + 1; j < i; j++) {
                if (block_rank[j*nt+i] == mype) {
                    num_comp_tasks++;
                    if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {
#ifdef USE_NANOS6
#pragma oss task depend(in: A[k][i], A[k][j]) depend(out: A[j][i]) firstprivate(k, j, i)
#else
#pragma omp task depend(in: A[k][i], A[k][j]) depend(out: A[j][i]) firstprivate(k, j, i)
#endif
{
						EXTRAE_ENTER(EVENT_GEMM);
			            START_TIMING(TIME_GEMM);
                        omp_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
			            END_TIMING(TIME_GEMM);
						EXTRAE_EXIT(EVENT_GEMM);
}
                    } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
#ifdef USE_NANOS6
#pragma oss task depend(in: C[i], A[k][j]) depend(out: A[j][i]) firstprivate(k, j, i)
#else
#pragma omp task depend(in: C[i], A[k][j]) depend(out: A[j][i]) firstprivate(k, j, i)
#endif
{
						EXTRAE_ENTER(EVENT_GEMM);
			            START_TIMING(TIME_GEMM);
                        omp_gemm(C[i], A[k][j], A[j][i], ts, ts);
			            END_TIMING(TIME_GEMM);
						EXTRAE_EXIT(EVENT_GEMM);
}
                    } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
#ifdef USE_NANOS6
#pragma oss task depend(in: A[k][i], C[j]) depend(out: A[j][i]) firstprivate(k, j, i)
#else
#pragma omp task depend(in: A[k][i], C[j]) depend(out: A[j][i]) firstprivate(k, j, i)
#endif
{
						EXTRAE_ENTER(EVENT_GEMM);
			            START_TIMING(TIME_GEMM);
                        omp_gemm(A[k][i], C[j], A[j][i], ts, ts);
			            END_TIMING(TIME_GEMM);
						EXTRAE_EXIT(EVENT_GEMM);
}
                    } else {
#ifdef USE_NANOS6
#pragma oss task depend(in: C[i], C[j]) depend(out: A[j][i]) firstprivate(k, j, i)
#else
#pragma omp task depend(in: C[i], C[j]) depend(out: A[j][i]) firstprivate(k, j, i)
#endif
{
						EXTRAE_ENTER(EVENT_GEMM);
			            START_TIMING(TIME_GEMM);
                        omp_gemm(C[i], C[j], A[j][i], ts, ts);
			            END_TIMING(TIME_GEMM);
						EXTRAE_EXIT(EVENT_GEMM);
}
                    }
                }
            }

            if (block_rank[i*nt+i] == mype) {
                num_comp_tasks++;
                if (block_rank[k*nt+i] == mype) {
#ifdef USE_NANOS6
#pragma oss task depend(in: A[k][i]) depend(out: A[i][i]) firstprivate(k, i)
#else
#pragma omp task depend(in: A[k][i]) depend(out: A[i][i]) firstprivate(k, i)
#endif
{
					EXTRAE_ENTER(EVENT_SYRK);
			        START_TIMING(TIME_SYRK);
                    omp_syrk(A[k][i], A[i][i], ts, ts);
			        END_TIMING(TIME_SYRK);
					EXTRAE_EXIT(EVENT_SYRK);
}
                } else {
#ifdef USE_NANOS6
#pragma oss task depend(in: C[i]) depend(out: A[i][i]) firstprivate(k, i)
#else
#pragma omp task depend(in: C[i]) depend(out: A[i][i]) firstprivate(k, i)
#endif
{
					EXTRAE_ENTER(EVENT_SYRK);
			        START_TIMING(TIME_SYRK);
                    omp_syrk(C[i], A[i][i], ts, ts);
			        END_TIMING(TIME_SYRK);
					EXTRAE_EXIT(EVENT_SYRK);
}
                }
            }
        }
    }
    END_TIMING(TIME_CREATE);
    }
#ifdef USE_NANOS6
#pragma oss taskwait
#else
#pragma omp taskwait
#endif
    END_TIMING(TIME_TOTAL);
    MPI_Barrier(MPI_COMM_WORLD);
#ifdef USE_TIMING
	PRINT_TIMINGS();
	FREE_TIMING();
#endif 
    printf("[%d] max_send_tasks %d, max_recv_tasks %d, num_send_tasks %d, num_recv_tasks %d, num_comp_tasks %d\n", 
           mype, max_send_tasks, max_recv_tasks, num_send_tasks, num_recv_tasks, num_comp_tasks);

    free(send_flags);

// pragma omp single
}// pragma omp parallel
}

