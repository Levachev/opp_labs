#include <pthread.h>
#include <stdio.h>
#include <queue>
#include <math.h>
#include <mpi.h>

#define LEN_TASK_Q 5
#define AMOUNT_OF_ITERATIONS 300
#define GIVE_TASK 1
#define GET_TASK 2
#define NO_TASK -3
#define END -4
 
pthread_t thrs[2];
pthread_mutex_t mutex;
int rank;
int size;
int loc_res;
int glob_res;
bool flag_of_end=false;

std::queue<int> tasks;

void input_tasks_queue(int iterCounter)
{
   int sum=0;
   pthread_mutex_lock(&mutex);         
   for(int i=0;i<LEN_TASK_Q;i++)
   {
      int tmp=abs(rank-(iterCounter%size))+3;
      sum+=tmp;
      tasks.push(tmp*10);
   }
   pthread_mutex_unlock(&mutex);
}

int execute_task(int wight)
{

   int res=0;
   for(int i=0;i<wight;i++)
   {
      res+=i;
   }
   return res;
}

void execute_tasks()
{
   while(true)
   {
      pthread_mutex_lock(&mutex);
      if(tasks.empty())
      {
         pthread_mutex_unlock(&mutex);
         return;
      }
      int tmp=tasks.front();
      tasks.pop();
      pthread_mutex_unlock(&mutex);
      loc_res+=execute_task(tmp); 
   }
}

void get_new_task(int *result)
{
   for(int i=0;i<size;i++)
   {
      if(i!=rank)
      {
         MPI_Send(&rank, 1, MPI_INT, i, GIVE_TASK, MPI_COMM_WORLD);
         
         MPI_Recv(result, LEN_TASK_Q+1, MPI_INT, i, GET_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

         if(result[0]!=NO_TASK)
         {
            return;
         }
      }
   }
   result[0]=NO_TASK;
}

void end_session()
{
   int end=END;
   MPI_Send(&end, 1, MPI_INT, rank, GIVE_TASK, MPI_COMM_WORLD);
}
 
void* execute(void *a)
{
   for(int i=0;i<AMOUNT_OF_ITERATIONS;i++)
   {
      input_tasks_queue(i);
      loc_res=0;
      execute_tasks();
      while(1)
      {
         int *recv_buf;
         recv_buf=(int*)calloc(LEN_TASK_Q+1, sizeof(int));

         get_new_task(recv_buf);

         if(recv_buf[0]==NO_TASK)
         {
            break;
         }
         else
         {
            pthread_mutex_lock(&mutex);
            for(int j=0;j<recv_buf[0];j++)
            {
               tasks.push(recv_buf[j+1]);
            }
            pthread_mutex_unlock(&mutex);
            execute_tasks();
         }
         free(recv_buf);
      }
      glob_res+=loc_res;
      MPI_Barrier(MPI_COMM_WORLD);
   }
   int main_result=0;
   MPI_Allreduce(&glob_res, &main_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   printf("main_result - %d\n", main_result);
   end_session();
   pthread_exit(nullptr);
}

int get_request()
{
    int request = 0;
    MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, GIVE_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return request;
}

void* handler(void *a)
{
   while(1)
   {
      int rank_request=get_request();

      if(rank_request==END)
      {
         printf("break\n");
         break;
      }
      int *send_buf;
      send_buf=(int*)calloc(LEN_TASK_Q+1, sizeof(int));

      pthread_mutex_lock(&mutex);      
      bool is_empty=tasks.empty();
      if(!is_empty)
      {
         int iter=0;
         while(!tasks.empty() && iter<2)
         {
            send_buf[iter+1]=tasks.front();
            tasks.pop();
            iter++;
         }
         pthread_mutex_unlock(&mutex);
         send_buf[0]=iter;
         MPI_Send(send_buf, LEN_TASK_Q+1, MPI_INT, rank_request, GET_TASK, MPI_COMM_WORLD);
      }
      else
      {
         pthread_mutex_unlock(&mutex);
         send_buf[0]=NO_TASK;
         MPI_Send(send_buf, LEN_TASK_Q+1, MPI_INT, rank_request, GET_TASK, MPI_COMM_WORLD);
      }
      free(send_buf);

   }
   pthread_exit(nullptr);
}
 
int main(int argc, char** argv)
{
   int provided;
   MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
   if(provided!=MPI_THREAD_MULTIPLE)
   {
      MPI_Finalize();
      return 0;
   }

   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   printf("size - %d rank - %d\n", size, rank);

   pthread_attr_t attrs;
 
   if(pthread_attr_init(&attrs))
   {
      perror("Cannot initialize attributes");
      return 0;
   }

   if(pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE))
   {
      perror("Error in setting attributes");
      return 0;
   }

   if(pthread_create(&thrs[0], &attrs, execute, nullptr))
   {
      perror("Cannot create a thread");
      return 0;
   }

   if(pthread_create(&thrs[1], &attrs, handler, nullptr))
   {
      perror("Cannot create a thread");
      return 0;
   }

   pthread_attr_destroy(&attrs);

   if(pthread_join(thrs[0], nullptr))
   {
      perror("Cannot join a thread");
      return 0;
   }
   if(pthread_join(thrs[1], nullptr))
   {
      perror("Cannot join a thread");
      return 0;
   }
   MPI_Finalize();
 
   return 0;
}
