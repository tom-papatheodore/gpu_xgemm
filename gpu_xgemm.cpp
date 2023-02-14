#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include "hip/hip_runtime.h"
#include <hipblas.h>
#include <rocblas.h>
#include <getopt.h>

/* ---------------------------------------------------------------------------------
Macro for checking errors in HIP API calls
----------------------------------------------------------------------------------*/
#define hipErrorCheck(call)                                                                \
do{                                                                                        \
    hipError_t hipErr = call;                                                              \
    if(hipSuccess != hipErr){                                                              \
      printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErr));  \
      exit(0);                                                                             \
    }                                                                                      \
}while(0)


/* ---------------------------------------------------------------------------------
Macro for checking success in hipBLAS API calls
----------------------------------------------------------------------------------*/
#define hipblasCheck(call)                                       \
do{                                                              \
  hipblasStatus_t hipblas_stat = call;                           \
  if(HIPBLAS_STATUS_SUCCESS != hipblas_stat){                    \
    std::cout << "hipblas call failed. Exiting..." << std::endl; \
    exit(1);                                                     \
  }                                                              \
}while(0)


/* ---------------------------------------------------------------------------------
Define default options
----------------------------------------------------------------------------------*/

// Size of arrays (default)
int N = 1024;

// Selector for sgemm (single) or dgemm (double) (default)
std::string precision = "double";


/* ---------------------------------------------------------------------------------
Parse command line arguments
----------------------------------------------------------------------------------*/
void print_help(){

    printf(
    "----------------------------------------------------------------\n"
    "Usage: ./gpu_xgemm [OPTIONS]\n\n"
    "--matrix_size=<value>, -m:       Size of matrices\n"
    "                                 (default is 1024)\n\n"
    " "
    "--precision=<value>,   -p:       <value> can be single or double\n"
    "                                 to select sgemm or dgemm\n"
    "                                 (default is double)\n\n"
    " "
    "--help,                -h:       Show help\n"
    "----------------------------------------------------------------\n"
    );
    exit(1);
}

void process_arguments(int argc, char *argv[]){

    const char* const short_options = "m:p:h";

    const option long_options[] = {
        {"matrix_size", optional_argument, nullptr, 'm'},
        {"precision",   optional_argument, nullptr, 'p'},
        {"help",        no_argument,       nullptr, 'h'},
        {nullptr,       no_argument,       nullptr,   0}
    };

    while(true){

        const auto opts = getopt_long(argc, argv, short_options, long_options, nullptr);

        if(-1 == opts){ break; }

        switch(opts){
            case 'm':
                N = std::stoi(optarg);
                break;
            case 'p':
                precision = std::string(optarg);
                break;
            case 'h':
            default:
                print_help();
                break;
        }
    }
}


/* ---------------------------------------------------------------------------------
Device xgemm wrappers - e.g., if you run with double precision, the double version 
will be used.
----------------------------------------------------------------------------------*/
hipblasStatus_t device_xgemm(hipblasHandle_t handle, double alpha, double beta, double *d_a, double *d_b, double *d_c){

    std::cout << "\nRunning hipblasDgemm...\n";
    return hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N,
                        &alpha, d_b, N, d_a, N, &beta, d_c, N);
}

hipblasStatus_t device_xgemm(hipblasHandle_t handle, float alpha, float beta, float *d_a, float *d_b, float *d_c){

    std::cout << "\nRunning hipblasSgemm...\n";
    return hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N,
                        &alpha, d_b, N, d_a, N, &beta, d_c, N);
}


/* ---------------------------------------------------------------------------------
Templated device xgemm test - e.g., if you run with double precision,
the type will be double (so read T as double).
----------------------------------------------------------------------------------*/
template <typename T>
T xgemm_test(T machine_eps){

    // Set which device to use
    int dev_id = 0;
    hipErrorCheck( hipSetDevice(dev_id) );

    // Set up for GPU timing
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float time_ms, time_ms_data_transfers, time_ms_gpu_compute;
    double total_time = 0.0;

    // Scaling factors
    T alpha = (T)1.0;
    T beta  = (T)0.0;

    // Size (in bytes) of individual arrays
    int buffer_size = N * N * sizeof(T);

    // Host matrix buffers
    T *A   = (T*)malloc(buffer_size);
    T *B   = (T*)malloc(buffer_size);
    T *C   = (T*)malloc(buffer_size);

    // Device matrix buffers
    T *d_A, *d_B, *d_C;
    hipErrorCheck( hipMalloc(&d_A, buffer_size) );
    hipErrorCheck( hipMalloc(&d_B, buffer_size) );
    hipErrorCheck( hipMalloc(&d_C, buffer_size) );

    // Fill matrices 
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {

            int index = i * N + j;

            if(j % 2 == 0){
                A[index] = sin(j);
            }
            else{
                A[index] = cos(j-1);
            }

            if(i % 2 == 0){
                B[index] = sin(i);
            }
            else{
                B[index] = cos(i-1);
            }

            C[index] = 0.0;
        }
    }

    // Pass host buffers to device buffers
    hipEventRecord(start, NULL);

    hipErrorCheck( hipMemcpy(d_A, A, buffer_size, hipMemcpyHostToDevice) );
    hipErrorCheck( hipMemcpy(d_B, B, buffer_size, hipMemcpyHostToDevice) );
    hipErrorCheck( hipMemcpy(d_C, C, buffer_size, hipMemcpyHostToDevice) );

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&time_ms_data_transfers, start, stop);

    // Create hipBLAS handle
    hipblasHandle_t handle;
    hipblasCheck( hipblasCreate(&handle) );

    // Call device_xgemm routine - e.g., if using double precision, the double-precision
    // version of the wrapper will be used.
    hipEventRecord(start, NULL);

    hipblasCheck( device_xgemm(handle, alpha, beta, d_A, d_B ,d_C) );

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&time_ms_gpu_compute, start, stop);

    // Copy results from device to host
    hipErrorCheck( hipMemcpy(C, d_C, buffer_size, hipMemcpyDeviceToHost) );

    // Make sure host and device found the same results
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){

            int index = i * N + j;
            T error = fabs( (C[index] - (T)N/(T)2.0) / ((T)N/(T)2.0) );

            if(error > machine_eps){
                std::cout << "\n!!!!!!!!!!!!!!!!!!!!"                    << std::endl;
                std::cout << "error = " << error << " > " << machine_eps << std::endl;
                std::cout << "Exiting..."                                << std::endl;
                std::cout << "!!!!!!!!!!!!!!!!!!!!"                      << std::endl;
                exit(1);
            }
        }
    }    

    // Clean up contexts and memory
    hipblasCheck( hipblasDestroy(handle) );

    hipErrorCheck( hipFree(d_A) );
    hipErrorCheck( hipFree(d_B) );
    hipErrorCheck( hipFree(d_C) );

    hipEventDestroy(start);
    hipEventDestroy(stop);

    free(A);
    free(B);
    free(C);

    double fp_ops    = (double)2.0 * (double)N * (double)N * (double)N;
    double gpu_flops = fp_ops / (time_ms_gpu_compute / 1000.0) ;

    std::cout << "\nGPU Data Transfers (s) : " << time_ms_data_transfers / 1000.0                         << std::endl;
    std::cout << "GPU Compute (s)        : "   << time_ms_gpu_compute / 1000.0                            << std::endl;
    std::cout << "Total GPU (s)          : "   << (time_ms_data_transfers + time_ms_gpu_compute) / 1000.0 << std::endl;
    std::cout << "GPU GFLOPS/s           : "   << gpu_flops / (double)(1000 * 1000 * 1000)                << std::endl;

    return gpu_flops / (double)(1000 * 1000 * 1000);

}


/* ---------------------------------------------------------------------------------
Main program
----------------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    process_arguments(argc, argv);

    rocblas_initialize();

    std::cout << "\n-----------------------------"   << std::endl;
    std::cout << "N = " << N                         << std::endl;
    std::cout << "precision = " << precision.c_str() << std::endl;
    std::cout << "-----------------------------"     << std::endl;

    int number_of_iterations = 11;
    double total_gflop_per_sec = 0.0;
    double gflop_per_sec       = 0.0;
    

    if(precision == "double"){

        for(int i=0; i<number_of_iterations; i++){

            double machine_epsilon = (double)2.23e-16;

            gflop_per_sec = xgemm_test<double>(machine_epsilon);

            if(i != 0){
                total_gflop_per_sec += gflop_per_sec;
            }
        }

    }
    else if(precision == "single"){

        for(int i=0; i<number_of_iterations; i++){

            float machine_epsilon = (float)1.20e-7;

            gflop_per_sec = xgemm_test<float>(machine_epsilon);

            if(i != 0){
                total_gflop_per_sec += gflop_per_sec;
            }
        }

    }
    else{
        std::cout << "Must choose either double or single for precision. Exiting..." << std::endl;
        exit(1);
    }

    std::cout << "\n***************************************"                                  << std::endl;
    std::cout << "Average GF/s: " << total_gflop_per_sec / (double)(number_of_iterations - 1) << std::endl;
    std::cout << "***************************************"                                    << std::endl;

    std::cout << "\n__SUCCESS__\n" << std::endl;    

    return 0;
}
