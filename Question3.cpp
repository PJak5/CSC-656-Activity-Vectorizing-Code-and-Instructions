#include <cstdio>

const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply";

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
   printf("Starting dgemv with n=%d\n", n);
   
   // For each row of A
   for(int i = 0; i < n; i++){
      // Initialize accumulator for dot product
      double dot_product = 0.0;
      
      // Calculate the starting index for this row
      const int row_idx = i * n;
      
      // Process in chunks of 4 doubles to encourage vectorization
      int j = 0;
      const int chunk_size = 4;
      
      if (n >= chunk_size) {
         double partial_sums[chunk_size] = {0.0};
         
         // Main vectorizable loop - process in chunks
         for (; j <= n - chunk_size; j += chunk_size) {
            // This loop can be auto-vectorized
            #pragma GCC ivdep
            for (int k = 0; k < chunk_size; k++) {
               partial_sums[k] += A[row_idx + j + k] * x[j + k];
            }
         }
         
         // Combine partial sums
         for (int k = 0; k < chunk_size; k++) {
            dot_product += partial_sums[k];
         }
      }
      
      // Handle remaining elements
      for (; j < n; j++) {
         dot_product += A[row_idx + j] * x[j];
      }
      
      // Update output vector
      y[i] += dot_product;
   }
   
   printf("Completed dgemv with n=%d\n", n);
}