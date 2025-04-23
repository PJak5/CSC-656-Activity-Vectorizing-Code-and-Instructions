#include <cstdio>
#include <omp.h>

const char* dgemv_desc = "Highly optimized DGEMV implementation with auto-vectorization and tiling.";

/*
This routine performs a dgemv operation
Y :=  A * X + Y
where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
On exit, A and X maintain their input values.
@param n       Problem size (dimension of square matrix)
@param A       Pointer to matrix data store in row-major order
@param x       Pointer to input vector
@param y       Pointer to output vector*/
void my_dgemv(int n, double* A, double* x, double* y) {
   printf("Starting optimized dgemv with n=%d\n", n);
   
   // Use OpenMP for parallelization
   #pragma omp parallel for if(n > 128)
   for(int i = 0; i < n; i++){
      // Calculate the starting index for this row
      const int row_idx = i * n;
      
      // Use multiple accumulators to reduce dependencies
      double sum0 = 0.0;
      double sum1 = 0.0;
      double sum2 = 0.0;
      double sum3 = 0.0;
      
      // Optimize for cache line utilization with loop unrolling
      // Process 16 elements at a time
      int j = 0;
      
      // Main vectorizable loop with unrolling
      for (; j <= n - 16; j += 16) {
         // First chunk of 4 elements
         #pragma GCC ivdep
         for (int k = 0; k < 4; k++) {
            sum0 += A[row_idx + j + k] * x[j + k];
         }
         
         // Second chunk of 4 elements
         #pragma GCC ivdep
         for (int k = 0; k < 4; k++) {
            sum1 += A[row_idx + j + 4 + k] * x[j + 4 + k];
         }
         
         // Third chunk of 4 elements
         #pragma GCC ivdep
         for (int k = 0; k < 4; k++) {
            sum2 += A[row_idx + j + 8 + k] * x[j + 8 + k];
         }
         
         // Fourth chunk of 4 elements
         #pragma GCC ivdep
         for (int k = 0; k < 4; k++) {
            sum3 += A[row_idx + j + 12 + k] * x[j + 12 + k];
         }
      }
      
      // Process remaining elements in chunks of 4
      for (; j <= n - 4; j += 4) {
         #pragma GCC ivdep
         for (int k = 0; k < 4; k++) {
            sum0 += A[row_idx + j + k] * x[j + k];
         }
      }
      
      // Handle remaining individual elements
      for (; j < n; j++) {
         sum0 += A[row_idx + j] * x[j];
      }
      
      // Combine all partial sums
      double dot_product = sum0 + sum1 + sum2 + sum3;
      
      // Update output vector
      y[i] += dot_product;
   }
   
   printf("Completed optimized dgemv with n=%d\n", n);
}