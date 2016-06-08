

/*
 * Document classification program
 */

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <ftw.h>

#define DICT_SIZE_MSG	0	/* Msg has dictionary size */
#define FILE_NAME_MSG	1	/* Msg is file name */
#define VECTOR_MSG	2	/* Msg is profile */ 
#define EMPTY_MSG	3	/* Msg is empty */

#define DIR_ARG		1	/* Directory argument */
#define DICT_ARG	2	/* Dictionary argument */
#define RES_ARG		3	/* Results argument */

typedef unsigned char uchar;

int main(int argc, char * argv[]) {

  int id;		/* Process rank */
  int p;		/* Number of processes */
  MPI_Comm worker_comm;	/*Workers-only communicator */
  
  void manager(int, char **, int);
  void worker(int, char **, MPI_Comm);
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  
  if(argc != 4) {
   if(!id) {
    printf("Program needs three arguments \n");
    printf("%s <dir> <dict> <results>\n", argv[0]);
   }
  } else if(p < 2) {
    printf("Program needs at least two processes\n");
  } else {
   if(!id) {
    MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, id, &worker_comm) ;
    manager(argc, argv, p);
   } else {
    MPI_Comm_split(MPI_COMM_WORLD, 0, id, &worker_comm);
    worker(argc, argv, worker_comm);
   }
  }
  MPI_Finalize();
  
  return 0;
}

void manager(int argc, char * argv[], int p) {
  int assign_cnt;	/* Docs assigned so far */ 
  int *assigned;		/* Document assignments */
  uchar *buffer;		/* Store profile vectors here */
  int dict_size;		/* Dictionary entries */
  int file_cnt;		/* Plain text files found */
  char **file_name;	/* Stores file (path) names */
  int i;
  MPI_Request pending;	/* Handle for recv request */
  int src;		/* Message source process */
  MPI_Status status;	/* Message status */	
  int tag;		/* Message tag */
  int terminated;	/* Count of terminated procs */
  uchar **vector;	/* Profile vector repository */
  
  void build_2d_array(int, int, uchar ***);
  void get_names(char *, char ***, int *);
  void write_profiles(char *, int, int, char **, uchar **);
  
  /* Put in request to receive dictionary size */
  MPI_Irecv(&dict_size, 1, MPI_INT, MPI_ANY_SOURCE, DICT_SIZE_MSG, MPI_COMM_WORLD, &pending);
  
  /* Collect the names of the documents to be profiled */
  get_names(argv[DIR_ARG], &file_name, &file_cnt);
  
  /* Wait for dictionary size to be received */
  MPI_Wait(&pending, &status);
  
  /* Set aside buffer to catch profiles from workers */
  buffer = (uchar *)malloc(dict_size * sizeof(MPI_UNSIGNED_CHAR));
  
  /* Set aside 2D array to hold all profiles
   * Call MPI_Abort if the allocation fails
   */
  build_2d_array(file_cnt, dict_size, &vector);
  
  /* Respond to requests by workers */
  terminated = 0;
  assign_cnt = 0;
  assigned = (int *)malloc(p * sizeof(int));
  
  do {
   /* Get profile from worker */
   MPI_Recv(buffer, dict_size, MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
   src = status.MPI_SOURCE;
   tag = status.MPI_TAG;
   if(tag == VECTOR_MSG) {
    for(i = 0; i < dict_size; i++)
      vector[assigned[src]][i] = buffer[i];
   }
   /* Assign more work or tell worker to stop */
   if(assign_cnt < file_cnt) {
    MPI_Send(file_name[assign_cnt], strlen(file_name[assign_cnt]) + 1, MPI_CHAR, src, FILE_NAME_MSG, MPI_COMM_WORLD);
    assigned[src] = assign_cnt;
    assign_cnt++;
   } else {
     MPI_Send(NULL, 0, MPI_CHAR, src, FILE_NAME_MSG, MPI_COMM_WORLD);
     terminated++;
   }
  } while(terminated < (p - 1));
  
  write_profiles(argv[RES_ARG], file_cnt, dict_size, file_name, vector);
}

void worker(int argc, char * argv[], MPI_Comm worker_comm)
{
  char *buffer;		/* Words in dictionary */
  hash_el **dict;	/* Hash table of words */	
  int dict_size;	/* Profile vector size */
  long file_len;	/* Chars in dictionary */
  int i;
  char *name;		/* Name of plain text files */
  int name_len;		/* Chars in file name */
  MPI_Request pending;	/* Handle for MPI_Send */
  uchar *profile;	/* Document profile vector */
  MPI_Status status;	/* Info about message */
  int worker_id;	/* Rank in worker_comm */
  
  void build_hash_table(char *, int, hash_el ***, int *);
  void make_profile(char *, hash_el **, int, uchar *);
  void read_dictionary(char *, char **, long *);
  
  /* Worker gets its worker ID number */
  MPI_Comm_rank(worker_comm, &worker_id);
  
  /* Worker makes intiial request for work */
  MPI_Isend(NULL, 0, MPI_UNSIGNED_CHAR, 0, EMPTY_MSG, MPI_COMM_WORLD, &pending);
  
  /* Read and broadcast dictionary file */
  if(!worker_id)
    read_dictionary(argv[DICT_ARG], &buffer, &file_len);
  MPI_Bcast(&file_len, 1, MPI_LONG, 0, worker_comm);
  if(worker_id) buffer = (char *)malloc(file_len);
  MPI_Bcast(buffer, file_len, MPI_CHAR, 0, worker_comm);
  
  /* Build hash table */
  build_hash_table(buffer, file_len, &dict, &dict_size);
  
  profile = (uchar *)malloc(dict_size * sizeof(uchar));
  
  /* Worker 0 sends msg to manager res size of dictionary */
  if(!worker_id) MPI_Send(&dict_size, 1, MPI_INT, 0, DICT_SIZE_MSG, MPI_COMM_WORLD);
  
  for(;;) {
   /*Find out length of file name */ 
   
   MPI_Probe(0, FILE_NAME_MSG, MPI_COMM_WORLD, &status);
   MPI_Get_count(&status, MPI_CHAR, &name_len);
   
   /* Drop out if no more work */
   if(!name_len) break;
   
   name = (char *)malloc(name_len);
   MPI_Recv(name, name_len, MPI_CHAR, 0, FILE_NAME_MSG, MPI_COMM_WORLD, &status);
   
   make_profile(name, dict, dict_size, profile);
   free(name);
   
   MPI_Send(profile, dict_size, MPI_UNSIGNED_CHAR, 0, VECTOR_MSG, MPI_COMM_WORLD);
  }
}


