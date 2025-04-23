//
// Copyright (C) 2025, E. Wes Bethel, All Rights Reserved.
// For educational use only.
// 

#include <iostream>
#include <stdio.h>

void
sum_rows(int N, int A[], int y[])
{
   // assume:
   // input array A[] is of size NxN, stored as a "flat, 1D array"
   // but we will consider A[] to be a logical, NxN 2D array consisting
   // of N rows and N columns

   // your job: write code that will for each row i in A, sum all the values 
   // all N columns of row row A[i,*] and place the sum into y[i]

   // Process each row
   for (int i = 0; i < N; i++) {
      y[i] = 0; // Initialize sum for this row
      
      // Use a pattern that's friendly to auto-vectorization
      const int chunk_size = 8;
      int j = 0;
      
      if (N >= chunk_size) {
         int partial_sums[chunk_size] = {0};
         
         // Process in chunks of 8 elements to encourage vectorization
         for (; j <= N - chunk_size; j += chunk_size) {
            // This loop can be auto-vectorized
            #pragma GCC ivdep
            for (int k = 0; k < chunk_size; k++) {
               partial_sums[k] += A[i*N + j + k];
            }
         }
         
         // Combine partial sums
         for (int k = 0; k < chunk_size; k++) {
            y[i] += partial_sums[k];
         }
      }
      
      // Handle remaining elements
      for (; j < N; j++) {
         y[i] += A[i*N + j];
      }
   }
}

int main(int ac, char*av[])
{
//   int N=(1<<10);
   int N=5;
   int A[N*N];
   int y[N];

   // initialize A
   for (int i=0;i<N*N;i++)
      A[i] = i%N;

   // initialize Y
   for (int i=0; i<N; i++)
      y[i] = -1 * i;


   // print out the A array
   printf(" Contents of the A[] array: \n");
   for (int indx=0,i=0;i<N;i++)
   {
      for (int j=0;j<N;j++)
         printf(" %d ", A[indx++]);
      printf("\n");
   }

   // print out the y vector
   printf(" Contents of the y[] vector: \n");
   for (int indx=0,i=0;i<N;i++)
   {
      printf(" %d ", y[indx++]);
   }
   printf("\n");

   sum_rows(N, A, y);

   // now do verification check
   int t=y[0], err=0, i;
   for (i=1; err == 0 && i<N ; i++)
      if (t != y[i])
      {
         err=1;
         printf(" correctness check fails at i=%d\n",i);
      }

   if (err==0)
      std::cout << "Correctness check succeeds " << std::endl;
}