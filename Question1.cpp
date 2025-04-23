//
// Copyright (C) 2025, E. Wes Bethel, All Rights Reserved.
// For educational use only.
// 
#include <iostream>

int 
sum_array(int N, int A[])
{
   // input consists of an integer N and an array A[] of size N
   // your job: write code to compute the sum of all values in A[]
   // and return that sum to the caller

   int sum = 0;
   
   // Use a loop structure that's friendly to compiler auto-vectorization
   // Process 8 elements at a time to encourage SIMD operations
   const int chunk_size = 8;
   int i = 0;
   
   // Handle main loop with vectorization-friendly pattern
   if (N >= chunk_size) {
       int partial_sums[chunk_size] = {0};
       
       for (; i <= N - chunk_size; i += chunk_size) {
           #pragma GCC ivdep
           for (int j = 0; j < chunk_size; j++) {
               partial_sums[j] += A[i + j];
           }
       }
       
       // Combine partial sums
       for (int j = 0; j < chunk_size; j++) {
           sum += partial_sums[j];
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