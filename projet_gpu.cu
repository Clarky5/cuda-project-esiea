
%%cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#define SIZE_ARRAY 400

//Fonction C du tri par insertion qui a été pris sur le site https://www.geeksforgeeks.org/insertion-sort/
void insertionSortCPU(int arr[], int n)
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

__global__ void tri_insertionGPU(int arr[], int sizeArray) 
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
            tid += blockIdx.x * blockDim.x;
        }
    }
}









int main(void)
{
    //On récupère les propriétés du périphérique GPU. 
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    
 
    int nThreads = properties.maxThreadsPerBlock; // On récupère le nombre maximum de thread par block
   
    int nBlocks = min((SIZE_ARRAY*SIZE_ARRAY + nThreads - 1) / nThreads,properties.maxGridSize[0]); // On récupère le nombre de blocks 

   


    int a[SIZE_ARRAY];
  
    int *dev_a;
 
    int i, j, tmp;
 
  
    //Evenement Cuda pour mesurer le temps d'exécution d'une fonction kernel
    cudaEvent_t start, stop;
    float gpu_time_used_ms;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    
    // Initialisation du générateur de nombres aléatoires
    srand(time(NULL));
    
   
    
    
 
    //allocation de la mémoire sur le GPU
    HANDLE_ERROR( cudaMalloc ( (void**)&dev_a, SIZE_ARRAY * sizeof(int) ) );
 
    printf("\nAffichage du tableau  avant le tri (tableau mélangé): \n");
 
    
    // Remplissage du tableau 
    for (i = 0; i < SIZE_ARRAY; i++) {
        a[i] = i*2;
    }
    

    

      // Mélange du tableau a
    for (i = 0; i < SIZE_ARRAY; i++) 
    {
        j = rand() % SIZE_ARRAY;
        tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }
 
 
    for (i = 0; i < SIZE_ARRAY; i++) 
    {
        printf("Index n° %d --> %d\n.",i,a[i]);
    }
    
   

    //On recopie le tableau du host vers le gpu
    HANDLE_ERROR( cudaMemcpy(dev_a, a, SIZE_ARRAY* sizeof(int), cudaMemcpyHostToDevice ) );
 

    

 
    tri_insertionGPU<<<nThreads,nBlocks>>>(dev_a, SIZE_ARRAY); //WARM-UP. Cela permet de réduire le temps d'exécution du kernel lorsque celle-ci est à nouveau mesuré.
 
    
    cudaEventRecord(start, 0);
    tri_insertionGPU<<<nThreads,nBlocks>>>(dev_a, SIZE_ARRAY);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time_used_ms, start, stop);
 


   clock_t cpu_start, cpu_end;
   double cpu_time_used_ms;

    cpu_start = clock();
    insertionSortCPU(a,SIZE_ARRAY);
    cpu_end = clock();

    cpu_time_used_ms = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000;

    


    
 
     printf("\n Affichage du tableau après le tri (CPU) : \n");
 
    
    for(int i = 0; i < SIZE_ARRAY; i++)
    {
        
        printf("Index n° %d --> %d\n.",i,a[i]);
    }

    //On recopie le tableau du gpu vers le host
    HANDLE_ERROR( cudaMemcpy( a, dev_a, SIZE_ARRAY* sizeof(int), cudaMemcpyDeviceToHost ) );
 
  
  

 
    printf("\n Affichage du tableau après le tri (GPU) : \n");
 
    
    for(int i = 0; i < SIZE_ARRAY; i++)
    {
        
        printf("Index n° %d --> %d\n.",i,a[i]);
    }

    
  printf("Temps d'execution CPU: %f ms\n", cpu_time_used_ms);
  printf("Temps d'execution GPU: %f ms\n", gpu_time_used_ms);
  
  
  
  //Synchronisation du kernel avec le CPU
  cudaDeviceSynchronize();

  //Libération de la mémoire allouée sur le GPU
  cudaFree(dev_a);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
 



 return 0;

}


