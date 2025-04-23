//
// Copyright (C) 2025, E. Wes Bethel, All Rights Reserved.
// For educational use only.
// 
#include <iostream>
#include <immintrin.h> // For AVX instructions

int 
sum_array(int N, int A[])
{
   // input consists of an integer N and an array A[] of size N
   // your job: write code to compute the sum of all values in A[]
   // and return that sum to the caller

   int sum = 0;
    
   int i = 0;
   if (N >= 8) {
       __m256i sum_vec = _mm256_setzero_si256();
        
       for (; i <= N - 8; i += 8) {
           __m256i v = _mm256_loadu_si256((__m256i*)&A[i]); // Load 8 integers
           sum_vec = _mm256_add_epi32(sum_vec, v); // Add to running sum
       }
        
       int temp[8];
       _mm256_storeu_si256((__m256i*)temp, sum_vec);
       for (int j = 0; j < 8; j++) {
           sum += temp[j];
       }
   }
    
   // Handle remaining elements
   for (; i < N; i++) {
       sum += A[i];
   }
    
   return sum;
}


int main(int ac, char*av[])
{
   int N=(1<<10);
   int A[N];

   // initialize A
   for (int i=0;i<N;i++)
      A[i] = i;

   int total = sum_array(N, A);

   std::cout << "Sum of all " << N << " items in A[] is: " << total << std::endl;

   if (total == (N*(N-1)/2))
      std::cout << "Correct check succeeds, you computed the correct answer." << std::endl;
   else
      std::cout << "Correct check fails, you didn't compute the correct answer." << std::endl;

}