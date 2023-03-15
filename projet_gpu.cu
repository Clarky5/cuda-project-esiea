%%cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curand_kernel.h>
#include <time.h>


//------------------------------------------------------------------------------------------------------------------------------
//Cette fonction est utilisée pour vérifier si l'appel à une fonction a crée une erreur.
//Si c'est le cas, alors on affiche un message d'erreur associé à cette erreur et on quitte le programme
//Cette fonction ne vient pas de moi. Elle a été récupéré sur le site https://stackoverflow.com/questions/13245258/handle-error-not-found-error-in-cuda 
//Elle est la réponse d'un utilisateur.

static void HandleError( cudaError_t err, const char *file,int line ) 
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        exit( EXIT_FAILURE );
    }
}


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//-------------------------------------------------------------------------------------------------------------------------------


//Taille du tableau
#define SIZE_ARRAY 500

//Fonction C du tri par insertion qui a été pris sur le site https://www.geeksforgeeks.org/insertion-sort/
void insertionSort(int arr[], int n)
{
    int i, key, j;
    for (i = 1; i < n; i++) {
        key = arr[i];
        j = i - 1;
 
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

//Fonction sur le gpu 

__global__ void tri_insertion_gpu(int *arr, int sizeArray) 
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
 
   int key, j;
   if (tid < sizeArray) 
   {
        for (tid = 1; tid < sizeArray; tid++) 
        {
            key = arr[tid];
            j = tid - 1;

            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }
            arr[j + 1] = key;
        }
    }
}






//Fonction remplissage sur le gpu
//On remplit le tableau directement depuis le GPU au lieu de le remplir 
__global__ void mixArray(int *a, curandState *state)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j, tmp;
    curand_init(clock64(), i, 0, &state[i]);

    if (i < SIZE_ARRAY) {
        a[i] = i * 2;
        
    }

    __syncthreads();

    for (i = SIZE_ARRAY - 1; i > 0; i--)
    {
        j = curand(&state[threadIdx.x]) % (i + 1);
        tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
}


int main(void)
{
    //On récupère les propriétés du périphérique GPU. 
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    
 
    int nThreads = properties.maxThreadsPerBlock; // On récupère le nombre maximum de thread par block
   
    int nBlocks = min((SIZE_ARRAY*SIZE_ARRAY + nThreads - 1) / nThreads,properties.maxGridSize[0]); // On récupère le nombre de blocks 

   


    int a[SIZE_ARRAY],b[SIZE_ARRAY];
  
    int *dev_a;
 
    int i, j, tmp;
 
  
    cudaEvent_t start, stop;
    float elapsed_time;
 
     // Allocation de mémoire pour les états CURAND. Utile pour générer des nombres aléatoires.
    curandState *dev_states;
    HANDLE_ERROR(cudaMalloc(&dev_states, SIZE_ARRAY * sizeof(curandState)));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    
    // Initialisation du générateur de nombres aléatoires
    srand(time(NULL));
    
   
    
    
 
    //allocation de la mémoire sur le GPU
    HANDLE_ERROR( cudaMalloc ( (void**)&dev_a, SIZE_ARRAY * sizeof(int) ) );
 
    printf("\nAffichage du tableau  avant le tri : \n");
 
    // Remplissage du tableau 
    for (i = 0; i < SIZE_ARRAY; i++) {
        a[i] = i * 2;
        b[i] = i * 2;
    }
 

      // Mélange du tableau a
    for (i = SIZE_ARRAY - 1; i > 0; i--) 
    {
        j = rand() % (i + 1);
        tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
 
 
    for (i = 0; i < SIZE_ARRAY; i++) 
    {
        printf("Index n° %d --> %d\n.",i,a[i]);
    }
    
    // Mélange du tableau b
    for (i = SIZE_ARRAY - 1; i > 0; i--) 
    {
        j = rand() % (i + 1);
        tmp = b[i];
        b[i] = b[j];
        b[j] = tmp;
    }
 





    HANDLE_ERROR( cudaMemcpy(dev_a, a, SIZE_ARRAY* sizeof(int), cudaMemcpyHostToDevice ) );
 

    

 
    tri_insertion_gpu<<<nThreads,nBlocks>>>(dev_a, SIZE_ARRAY); //WARM-UP. Cela permet de réduire le temps d'exécution de la fonction lorsque celle-ci est à nouveau mesuré.
 
    
    cudaEventRecord(start, 0);
    tri_insertion_gpu<<<nThreads,nBlocks>>>(dev_a, SIZE_ARRAY);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);
 


   clock_t cpu_start, cpu_end;
   double cpu_time_used_ms;

    cpu_start = clock();
    insertionSort(a,SIZE_ARRAY);
    cpu_end = clock();

    cpu_time_used_ms = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000;

    


    insertionSort(a,SIZE_ARRAY);


    HANDLE_ERROR( cudaMemcpy( a, dev_a, SIZE_ARRAY* sizeof(int), cudaMemcpyDeviceToHost ) );
 
  
  

 
    printf("\n Affichage du tableau après le tri : \n");
 
    
    for(int i = 0; i < SIZE_ARRAY; i++)
    {
        
        printf("Index n° %d --> %d\n.",i,a[i]);
    }

    
  printf("Temps d'execution CPU: %f ms\n", cpu_time_used_ms);
  printf("Temps d'execution GPU: %f ms\n", elapsed_time);
  
  
  
  cudaDeviceSynchronize();

  //Libération de la mémoire allouée sur le GPU
  cudaFree(dev_a);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
 



 return 0;

}


