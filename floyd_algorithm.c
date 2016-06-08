/* The Floyd's algorithm (all-pairs shortest-path problem), Version 1
 * 
 * The sequential algorithm is as follows:
 * Input:	n - number of vertices
 * 		a[0...n-1, 0...n-1] - adjacency matrix
 * Output:	Transformed a that contains the shortest path lengths
 * 
 * for k <- 0 to n-1
 * 	for i <- 0 to n - 1
 * 		for j <- 0 to n - 1
 * 			a[i,j] <- min(a[i,j],a[i,k] + a[k,j])
 *		endfor
 * 	endfor
 * endfor
 * 
 * Time Complexity - O(n^3)
 * 
 * Author: Michael Quinn
 * 
 * Last modification: 12 May 2016
 * 
 */

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include "helpersMPI.h"

typedef int dtype;
#define MPI_TYPE MPI_INT

int main(int argc, char * argv[]) {
  dtype** a;		/* Doubly-subscripted array */
  dtype* storage;	/* Local portion of array elements */
  int i, j, k;		/* Loop counters */
  int id;		/* Process rank */
  int m;		/* Rows in matrix */
  int n;		/* Columns in matrix */
  int p;		/* Number of processes */
  
  m = n = 0;
  
  void compute_shortest_paths(int, int, int**, int);
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  read_row_striped_matrix(argv[1], (void *)&a, (void *)&storage,
			  MPI_TYPE, &m, &n, MPI_COMM_WORLD);
  
  if(m != n) terminate(id, "Matrix must be square\n");
  
  print_row_striped_matrix((void **)a, MPI_TYPE, m, n,
    MPI_COMM_WORLD);
  compute_shortest_paths(id, p, (dtype **)a, n);
  print_row_striped_matrix((void **)a, MPI_TYPE, m, n,
    MPI_COMM_WORLD);
  
  MPI_Finalize();
  
  exit(0);
}

void compute_shortest_paths(int id, int p, dtype **a, int n) {
  
  int i, j, k;
  int offset;		/* Local index of broadcast row */
  int root;		/* Process controlling row to be bcast */
  int *tmp;		/* Holds the broadcast row */
  
  
  tmp = (dtype *)malloc(n * sizeof(dtype));
  for(k = 0; k < n; k++) {
    root = BLOCK_OWNER(k, p, n);
    if(root == id) {
      offset = k * BLOCK_LOW(id, p, n);
      for(j = 0; j < n; j++)
	tmp[j] = a[offset][j];
    }
    MPI_Bcast(tmp, n, MPI_TYPE, root, MPI_COMM_WORLD);
    for(i = 0; i < BLOCK_SIZE(id, p, n); i++)
      for(j = 0; j < n; j++)
	a[i][j] = MIN(a[i][j], a[i][k] + tmp[j]);
  }
  free(tmp);
}

