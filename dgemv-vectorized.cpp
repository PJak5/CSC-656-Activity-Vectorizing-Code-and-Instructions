#include <cstdio>

const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 * @param n       Problem size (dimension of square matrix)
 * @param A       Pointer to matrix data store in row-major order
 * @param x       Pointer to input vector
 * @param y       Pointer to output vector
 */
void my_dgemv(int n, double* A, double* x, double* y) {
   // insert your code here: implementation of vectorized vector-matrix multiply
   printf("Starting dgemv with n=%d\n", n);
   for(int i = 0; i < n; i++){
      // Initialize accumulator for dot product
      double dot_product = 0.0;

      // Calculate the starting index for this row
      const int row_idx = i * n;

      // Compute dot product of row i with vector x
      for (int j = 0; j < n; j++){
         // Access memory with unit stride for better cache performance
         dot_product += A[row_idx + j] * x[j];
      }

      // Update output vector
      y[i] += dot_product;
   }
   printf("Completed dgemv with n=%d\n", n);

}
