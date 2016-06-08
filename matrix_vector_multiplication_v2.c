/* Vector-matrix multiplication, Version 2
 * 
 * The sequential algorithm is as follows:
 * Input:	a[0...m - 1,0...n - 1] - matrix with dimensions m x n
 * 		b[0...n - 1] - vector with dimensions n x 1
 * Output:	c[0...m - 1] - vector with dimensions m x 1
 * 
 * for i <- 0 to m - 1
 * 	c[i] <- 0
 * 	for j <- 0 to n - 1
 *		c[i] <- c[i] + a[i][j] x b[j] 
 * 	endfor
 * endfor
 * 
 * Time Complexity - O(mn)
 * 
 * Author: Michael Quinn
 * 
 * Last modification: 16 May 2016
 * 
 */

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include "helpersMPI.h"

/* Change these two definitions when the matrix and vector  element types changes */

typedef double dtype;
#define mpitype MPI_DOUBLE

int main(int argc, char * argv[]) {

  dtype **a;		/* First factor, a matrix */
  dtype *b;		/* Second factor, a vector */
  dtype *c_block;	/* Partial product vector */
  dtype *c;		/* Replicated product vector */
  dtype *c_part_out;	/* Partial sums sent */
  dtype *c_part_in;	/* Partial sums to each proc */
  int *cnt_out;		/* Elements sent to each proc */
  int *cnt_in;		/* Elements received per proc */
  int *disp_out;	/* Indices of sent elements */
  int *disp_in;		/* Indices of received elements */
  dtype *storage;	/* Matrix elements stored here */
  int i, j;		/* Loop indices */
  int id; 		/* Process ID number */
  int local_els;	/* Cols of 'a' and elements of 'b' held by this process */
  int m;		/* Rows in matrix */
  int n;		/* Columns in matrix */
  int nprime;		/* Elements in vector */
  int p;		/* Number of processes */
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  read_col_striped_matrix(argv[1], (void ***)&a, (void **)&storage, mpitype, &m, &n, MPI_COMM_WORLD);
  print_col_striped_matrix((void **)a, mpitype, m, n, MPI_COMM_WORLD);

  read_block_vector(argv[2], (void **) &b, mpitype, &nprime, MPI_COMM_WORLD);
  print_block_vector((void *)b, mpitype, nprime, MPI_COMM_WORLD);
  
  /* Each process multiplies its columns of 'a' and vector 
   * 'b' resulting in a partial sum of product 'c'
   */
  c_part_out = (dtype *)my_malloc(id, n * sizeof(dtype));
  local_els = BLOCK_SIZE(id, p, n);
  
  for(i = 0; i < n; i++) {
    c_part_out[i] = 0.0;
    for(j = 0; j < local_els; j++)
      c_part_out[i] += a[i][j] * b[j];
  }
  
  create_mixed_xfer_arrays(id, p, n, &cnt_out, &disp_out);
  create_uniform_xfer_arrays(id, p, n, &cnt_in, &disp_in);
  c_part_in = (dtype *)my_malloc(id, p * local_els * sizeof(dtype));
  
  MPI_Alltoallv(c_part_out, cnt_out, disp_out, mpitype, c_part_in, cnt_in, disp_in, mpitype, MPI_COMM_WORLD);
  
  c = (dtype *)my_malloc(id, local_els * sizeof(dtype));
  for(i = 0; i < local_els; i++) {
   c[i] = 0.0;
   for(j = 0; j < p; j++)
     c[i] += c_part_in[i + j * local_els];
  }
  
  print_block_vector((void *)c, mpitype, n, MPI_COMM_WORLD);
  MPI_Finalize();
  
  return 0;
}

