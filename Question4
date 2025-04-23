#include <cstdio>
#include <immintrin.h> // For AVX instructions

const char* dgemv_desc = "Highly optimized vectorized implementation of matrix-vector multiply.";

void my_dgemv(int n, double* A, double* x, double* y) {
   // For each row of A
   #pragma omp parallel for if(n > 128)
   for(int i = 0; i < n; i++){
      // Calculate the starting index for this row
      const int row_idx = i * n;
      
      // Initialize multiple accumulators for better instruction-level parallelism
      __m256d sum0 = _mm256_setzero_pd();
      __m256d sum1 = _mm256_setzero_pd();
      __m256d sum2 = _mm256_setzero_pd();
      __m256d sum3 = _mm256_setzero_pd();
      
      // Loop unrolling - process 16 elements (4 vectors) at a time
      int j = 0;
      for (; j <= n - 16; j += 16) {
         // First chunk of 4 doubles
         __m256d a_vec0 = _mm256_loadu_pd(&A[row_idx + j]);
         __m256d x_vec0 = _mm256_loadu_pd(&x[j]);
         #ifdef __FMA__
         sum0 = _mm256_fmadd_pd(a_vec0, x_vec0, sum0); // FMA: multiply and add
         #else
         sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(a_vec0, x_vec0));
         #endif
         
         // Second chunk of 4 doubles
         __m256d a_vec1 = _mm256_loadu_pd(&A[row_idx + j + 4]);
         __m256d x_vec1 = _mm256_loadu_pd(&x[j + 4]);
         #ifdef __FMA__
         sum1 = _mm256_fmadd_pd(a_vec1, x_vec1, sum1);
         #else
         sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(a_vec1, x_vec1));
         #endif
         
         // Third chunk of 4 doubles
         __m256d a_vec2 = _mm256_loadu_pd(&A[row_idx + j + 8]);
         __m256d x_vec2 = _mm256_loadu_pd(&x[j + 8]);
         #ifdef __FMA__
         sum2 = _mm256_fmadd_pd(a_vec2, x_vec2, sum2);
         #else
         sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(a_vec2, x_vec2));
         #endif
         
         // Fourth chunk of 4 doubles
         __m256d a_vec3 = _mm256_loadu_pd(&A[row_idx + j + 12]);
         __m256d x_vec3 = _mm256_loadu_pd(&x[j + 12]);
         #ifdef __FMA__
         sum3 = _mm256_fmadd_pd(a_vec3, x_vec3, sum3);
         #else
         sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(a_vec3, x_vec3));
         #endif
      }
      
      // Process remaining elements in chunks of 4
      for (; j <= n - 4; j += 4) {
         __m256d a_vec = _mm256_loadu_pd(&A[row_idx + j]);
         __m256d x_vec = _mm256_loadu_pd(&x[j]);
         #ifdef __FMA__
         sum0 = _mm256_fmadd_pd(a_vec, x_vec, sum0);
         #else
         sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(a_vec, x_vec));
         #endif
      }
      
      // Combine the partial sums
      sum0 = _mm256_add_pd(sum0, sum1);
      sum2 = _mm256_add_pd(sum2, sum3);
      sum0 = _mm256_add_pd(sum0, sum2);
      
      // Horizontal sum of the 4 elements in sum0
      __m128d sum_high = _mm256_extractf128_pd(sum0, 1);
      __m128d sum_low = _mm256_castpd256_pd128(sum0);
      __m128d sum_hl = _mm_add_pd(sum_high, sum_low);
      __m128d sum_lh = _mm_permute_pd(sum_hl, 1);
      __m128d result = _mm_add_sd(sum_hl, sum_lh);
      double dot_product = _mm_cvtsd_f64(result);
      
      // Handle remaining elements
      for (; j < n; j++) {
         dot_product += A[row_idx + j] * x[j];
      }
      
      // Update output vector
      y[i] += dot_product;
   }
}