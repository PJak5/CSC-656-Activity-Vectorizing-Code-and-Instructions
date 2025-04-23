#include <cstdio>

const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

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
   
   for(int i = 0; i < n; i++){
      const int row_idx = i * n;
      
      __m256d sum_vec = _mm256_setzero_pd();
      
      int j = 0;
      for (; j <= n - 4; j += 4) {
         __m256d a_vec = _mm256_loadu_pd(&A[row_idx + j]);
         __m256d x_vec = _mm256_loadu_pd(&x[j]);
         
         __m256d prod = _mm256_mul_pd(a_vec, x_vec);
         sum_vec = _mm256_add_pd(sum_vec, prod);
      }
      
      double temp[4];
      _mm256_storeu_pd(temp, sum_vec);
      double dot_product = temp[0] + temp[1] + temp[2] + temp[3];
      
      for (; j < n; j++) {
         dot_product += A[row_idx + j] * x[j];
      }
      
      y[i] += dot_product;
   }
   
   printf("Completed dgemv with n=%d\n", n);
}
