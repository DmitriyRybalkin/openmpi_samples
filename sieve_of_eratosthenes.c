
/* The Sieve or Eratosthenes, Version 1
 * 
 * The sequential algorithm is as follows:
 * 1. Create a list of natural numbers 2, 3, 4, ..., n none of which is marked
 * 2. Set k to 2, the first unmarked number on the list
 * 3. Repeat:
 * 	(a) Mark all multiples of k between k^2 and n
 * 	(b) Find the smallest number greater than k
 * 		that is unmarked. Set k to this new value
 * 	Until k^2 > n
 * 4. The unmarked numbers are primes
 * 
 * Author: Michael Quinn
 * 
 * Last modification: 12 May 2016
 * 
 * Time complexity:
 * X(n ln ln n)/p + (sqrt(n)/ln sqrt(n))l[log p]
 * 
 * Where:
 * X - time needed to mark a particular cell as being multiple of a prime
 * O(n ln ln n) 	- complexity of sequential algorithm
 * l 			- message latency
 * l[log p] 		- cost of each data broadcast
 * [log p] 		- ceiling of log p
 * n/ln n 		- the number of primes between 2 and n
 * sqrt(n)/ln sqrt(n) 	- approximation to the number of loop iterations
 */

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "helpersMPI.h"

int main(int argc, char * argv[]) {

  int count;		/* local prime count */
  double elapsed_time;	/* parallel execution time */
  int first;		/* index of first multiple */
  int global_count;	/* global prime count */
  int high_value;	/* highest value on this proc */
  int i;		/* loop counter */
  int id;		/* process id number */
  int index;		/* index of current prime */
  int low_value;	/* lowest value on this proc */
  char * marked;	/* portion of 2,...,'n' */
  int n;		/* sieving from 2,..., 'n' */
  int p;		/* number of processes */
  int proc0_size;	/* size of proc 0's subarray */
  int prime;		/* current prime */
  int size;		/* elements in 'marked' */
  
  MPI_Init(&argc, &argv);
  
  /* Start the timer */
  MPI_Barrier(MPI_COMM_WORLD);
  elapsed_time =- MPI_Wtime();
  
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  if(argc != 2) {
   if(!id) printf("Command line: %s <m>\n", argv[0]);
   MPI_Finalize();
   exit(1);
  }
  
  n = atoi(argv[1]);
  
  /* Figure out this process's share of the array,
   * as well as the integers represented by the first
   * and last array elements
   */
  low_value = 2 + BLOCK_LOW(id, p, n - 1);
  high_value = 2 + BLOCK_HIGH(id, p, n - 1);
  size = BLOCK_SIZE(id, p, n - 1);
  
  /* Bail out if all the primes used for sieving 
   * are not all held by process 0
   */
  proc0_size = (n - 1)/p;
  
  /* Algorithm works only if the square of the largest value
   * in process 0's array is greater than the upper limit of the
   * sieve.
   */
  if((2 + proc0_size) < (int) sqrt((double)n)) {
   if(!id) printf("Too many processes\n") ;
   MPI_Finalize();
   exit(1);
  }
  
  /* Allocate the process's share of the array 
   * Single byte is the smallest unit if memory
   * that can be indexed in C, so declare the array
   * to be of type char.
   */
  marked = (char *) malloc(size);
  
  if(marked == NULL) {
   printf("Cannot allocate enough memory\n");
   MPI_Finalize();
   exit(1);
  }
  
  /* Unmark list of elements */
  for(i = 0; i < size; i++) marked[i] = 0;
  if(!id) index = 0;
  prime = 2;
  
  do {
    if(prime * prime > low_value)
      first = prime * prime - low_value;
    else {
      if(!(low_value % prime)) first = 0;
      else first = prime - (low_value % prime);
    }
    /* Start actual sieving 
     * Each process marks the multiples
     * of the current prime number from the
     * first index through the end of the
     * array.
     */
    for(i = first; i < size; i += prime) marked[i] = 1;
    
    /* Process 0 finds the next prime by locating 
     * the next unamrked location in the array.
     */
    if(!id) {
     while (marked[++index]);
     prime = index + 2;
    }
    /* Process 0 broadcasts the value of the next prime
     * to the other processes
     * 
     * int MPI_Bcast (
     * 	void *buffer,		// Addr of 1st broadcast element
     * 	int count,		// Number of elements to broadcast
     * 	MPI_Datatype datatype,	// Type of elements to broadcast
     * 	int root,		// ID of process doing broadcast
     * 	MPI_Comm comm		// Communicator
     * )
     */
    MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
    /* The process continue to sieve as long as the square
     * of the current prime is less than or equal to the
     * upper limit
     */
  } while(prime * prime <= n);
  
  /* Each process counts the number of primes in its portion
   * of the list.
   */
  count = 0;
  for(i = 0; i < size; i++)
    if(!marked[i]) count++;
    
  /* The process compute the grand total
   * with the result being stored in variable
   * global_count on process 0
   */
  MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  
  /* Stop the timer */
  elapsed_time += MPI_Wtime();
  
  /* Print the results 
   * Process 0 prints the answer and elapsed time
   */
  if(!id) {
   printf("%d primes are less than or equal to %d\n", global_count, n);
   printf("Total elapsed time: %10.6f\n", elapsed_time);
  }
  
  MPI_Finalize();
  
  exit(0);
}