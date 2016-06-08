
#include <mpi.h>
#include <stdio.h>

/*
 * Circuit Satisfiability, Version 3
 * 
 * This MPI program determines whether a cricuit is
 * satisfiable, that is, whether there is a combination of
 * inputs that causes the output of the circuit to be 1.
 * The particular circuit being tested is 'wired' into the
 * logic of function 'check_circuit'. All combinations of
 * inputs that satisfy the circuit are printed.
 * 
 * Programmed by Dmitriy Rybalkin <dmitriy.rybalkin@gmail.com>
 * 
 * Last modification: 11 May 2016
 * 
 * This is enhanced version of the program also prints the
 * total number of solutions
 * 
 * Also amended with elapsed time measuring functions
 * double MPI_Wtime, double MPI_Wtick. Commented out printf and fflush within function check_circuit to avoid counting I/O time
 */

/* Return 1 if 'i'th bit of 'n' is 1; 0 otherwise */
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)

int check_circuit(int id, int z) {
  int v[16];	/* Each element is a bit of 'z' */
  int i;
  int count_solution = 0;
  
  for(i = 0; i < 16; i++) v[i] = EXTRACT_BIT(z, i);
  
  if((v[0] || v[1]) && (!v[1] || !v[3]) && (v[2] || v[3])
    && (!v[3] || !v[4]) && (v[4] || !v[5])
    && (v[5] || !v[6]) && (v[5] || v[6])
    && (v[6] || !v[15]) && (v[7] || !v[8])
    && (!v[7] || !v[13]) && (v[8] || v[9])
    && (v[8] || !v[9]) && (!v[9] || !v[10])
    && (v[9] || v[11]) && (v[10] || v[11])
    && (v[12] || v[13]) && (v[13] || !v[14])
    && (v[14] || v[15])) {
      /*printf("%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", id,
	v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],
	v[10],v[11],v[12],v[13],v[14],v[15]);
      fflush(stdout);*/
      count_solution++;
    }
    return count_solution;
}

int main(int argc, char * argv[]) {

  int global_solutions;	/* Total number of solutions */
  int i;
  int id;		/* Process rank */
  int p;		/* Number of processes */
  int solutions;	/* Solutions found by this proc */
  int check_circuit(int, int);
  double elapsed_time;
  
  /* call this before any other MPI functions */
  MPI_Init(&argc, &argv);
  /* barrier sync. to measure elapsed time after all executions reach the barrier point */
  MPI_Barrier(MPI_COMM_WORLD);
  elapsed_time =- MPI_Wtime();
  
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  solutions = 0;
  
  for(i = id; i < 65536; i += p)
   solutions += check_circuit(id, i); 
  
  /*
   * Function MPI_Reduce performs
   * one or more reduction operations
   * on values submitted by all processes in
   * a communicator. The header is:
   * int MPI_Reduce (
   * 	void *operand		//addr of 1st reduction element
   * 	void *result		//addr of 1st reduction result
   * 	int count		//reductions to perform
   * 	MPI_Datatype type	//type of elements
   * 	MPI_Op operator		//reduction operator
   * 	int root		//process getting results
   * 	MPI_Comm comm		//communicator
   * )
   */
  MPI_Reduce(&solutions, &global_solutions, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  elapsed_time += MPI_Wtime();
  
  printf("Process %d is done\n", id);
  fflush(stdout);
  
  MPI_Finalize();
  
  printf("Elapsed time is %f\n", elapsed_time);
  fflush(stdout);
  
  if(id == 0) printf("There are %d different solutions\n", global_solutions);
  
  return 0;
}