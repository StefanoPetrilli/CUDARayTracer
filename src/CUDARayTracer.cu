#include <stdio.h>
#include "imageGenerator.h"
#include "constants.h"
#include "float3.h"
#include "ray.h"
#include "hitable.h"

//TODO trovare il motivo del comportamento inaspettato
//TODO calcolare la normale solo del punto più vicino
//TODO scrivere i commenti sulla sfera


//--Optional
////TODO Fare in modo che sia possibile costruire scene dinamicamente e gli oggetti delle scene
//				che vengono costruite verranno salvati nella memoria statica

void gpuErrorCheck(int i = 0){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		    printf("Error %d: %s\n", i, cudaGetErrorString(err));
}

/**
 * Given a ray, create the blend of colors
 */
__device__ float3 color(ray &r, hitable *world){
	hitRecord rec;

	if (world->hit(r, 0.0, MAXFLOAT, rec)) {
		return 0.5 * make_float3(rec.normal.x + 1.0, rec.normal.y + 1.0, rec.normal.z + 1.0);
	} else {
		float y = unitVector(r.direction()).y;
		float t = 0.5 * (y + 1.0);
		return (1.0 - t) * make_float3(1.0, 1.0, 1.0) + t * make_float3(0.5, 0.7, 1.0);
	}
}

/**
 * The nvidia's GPUs have 64kb of memory that can be filled with constant variables.
 * Using this memory in the raytracer gives two advantages:
 * 1 - Sligly increase the performance because it is cached
 * 2 - Sligly increase the performance because those variables are allocated only once for
 * 		thousands of cores
 * 	3- Once again, since they are allocated once for all the cores a noticable amount of memory
 * 		remain free
 */
__constant__ float3 lowerLeftCorner;
__constant__ float3 vertical;
__constant__ float3 origin;
__constant__ float3 horizontal;

__global__ void kernel(int* imageGpu){

	hitable *list[2];

	list[0] = new sphere(make_float3(0, 0, -1), 0.5);
	list[1] = new sphere(make_float3(10, -100.5, -1), 100);
	hitable *world = new hitableList(list, 2);

	//the variable that increases is the h but the i keep the count of time of execution
	//same with the w, but the j keep the counting
	for (int i = 0, h = blockIdx.x; i < HEIGHT; i++) {
		if (h >= HEIGHT) continue;
		for (int j = 0, w = threadIdx.x; j < WIDTH; j++) {
			if (w >=  WIDTH) continue;

			//Those are relative coordinates based on the virtual camera
			float u =  float(w) / float(WIDTH);
			float v = float(h)  / float(HEIGHT);


			//Calculate the color of the ray based on the relative coordinates
			ray r(origin, lowerLeftCorner + (u * horizontal) + (v * vertical));
			float3 col = color(r, world);

			//printf("%d\n", int(255.99 * col.x));

			imageGpu[addressConverter(h, w, 0)] = int(255.99 * col.x);
			imageGpu[addressConverter(h, w, 1)] = int(255.99 * col.y);
			imageGpu[addressConverter(h, w, 2)] = int(255.99 * col.z);

			if(imageGpu[addressConverter(h, w, 0)] > 256 || imageGpu[addressConverter(h, w, 0)] < 0) printf("Errore");
			if(imageGpu[addressConverter(h, w, 1)] > 256 || imageGpu[addressConverter(h, w, 1)] < 0) printf("Errore");
			if(imageGpu[addressConverter(h, w, 2)] > 256 || imageGpu[addressConverter(h, w, 2)] < 0) printf("Errore");

			w = threadIdx.x + (blockDim.x * j);
		}
		h = blockIdx.x + (gridDim.x * i);

	}

}

int main ()
{

	//Allocate a tridimensional vector that contains the image's data
	int *image = (int*) malloc(sizeof(int) * WIDTH * HEIGHT * BYTESPERPIXEL);
	memset((int*) image, 1, sizeof(int) * WIDTH * HEIGHT * BYTESPERPIXEL);

	//Initialize the data that will reside in the constant gpu memory in the
	//TODO capire che vogliono dire queste variabili
	float3 lowerLeftCornerCPU = make_float3(-2.0, -1.0, -1.0);
	float3 horizontalCPU = make_float3(4.0, 0.0, 0.0);
	float3 verticalCPU = make_float3(0.0, 2.0, 0.0);
	float3 originCPU = make_float3(0.0, 0.0, 0.0);

	//Send the data to the constant memory
	cudaMemcpyToSymbol(lowerLeftCorner, &lowerLeftCornerCPU, sizeof(float3), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(horizontal, &horizontalCPU, sizeof(float3));
	cudaMemcpyToSymbol(vertical, &verticalCPU, sizeof(float3));
	cudaMemcpyToSymbol(origin, &originCPU, sizeof(float3));
	gpuErrorCheck();

	//deallocate the constant data that now reside in the constant memory from the cpu
	//TODO deallocate i float3 che ora sono in memoria costante

	//Allocate the memory on the gpu
	int *imageGpu;
	cudaMalloc((void**) &imageGpu, sizeof(int) * WIDTH * HEIGHT * BYTESPERPIXEL);
	gpuErrorCheck();

	//Transfer the image to the gpu for the elaboration
	cudaMemcpy(imageGpu, image, sizeof(int) * WIDTH * HEIGHT * BYTESPERPIXEL, cudaMemcpyHostToDevice);
	gpuErrorCheck();

	kernel<<<BLKSIZE, THRDSIZE>>>(imageGpu);
	gpuErrorCheck();

	//Take back the image
	cudaMemcpy(image, imageGpu,  sizeof(int) * WIDTH * HEIGHT * BYTESPERPIXEL, cudaMemcpyDeviceToHost);
	gpuErrorCheck();

	generate(WIDTH, HEIGHT, BYTESPERPIXEL, image);

	printf("fine\n");
    return EXIT_SUCCESS;
}
