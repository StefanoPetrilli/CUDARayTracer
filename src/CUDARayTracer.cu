#include <stdio.h>
#include "imageGenerator.h"
#include "constants.h"
#include "float3.h"
#include "ray.h"
#include "hitable.h"
#include <curand.h>
#include <curand_kernel.h>


//TODO generate numberm more randomly
//TODO eliminate the recursion
//TODO write some comments

//--Optional
//TODO make the antialiasing sampling random.
////TODO Fare in modo che sia possibile costruire scene dinamicamente e gli oggetti delle scene
//				che vengono costruite verranno salvati nella memoria statica

/**
 * The nvidia's GPUs have 64kb of memory that can be filled with constant variables.
 * Using this memory in the raytracer gives two advantages:
 * 1 - Sligly increase the performance because it is cached
 * 2 - Sligly increase the performance because those variables are allocated only once for
 * 		thousands of cores
 * 	3- Once again, since they are allocated once for all the cores a noticable amount of memory
 * 		remain free
 */
//These are used to represent the camera
__constant__ float3 lowerLeftCorner;
__constant__ float3 vertical;
__constant__ float3 origin;
__constant__ float3 horizontal;
__constant__ sphere spheres[OBJNUMBER];

//Number of spheres
const int sn = 2;

__device__ curandState_t state;

void gpuErrorCheck(int i = 0){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		    printf("Error %d: %s\n", i, cudaGetErrorString(err));
}

//TODO pass the actual time to make it random
__device__ int pw = 0;
__device__ void rndInit(){

	pw = pw + threadIdx.x  + blockIdx.x * blockDim.x;
	//Init the parameters used to generate the random number
	curand_init(pw, 0, 0, &state);
}

__device__ float getRndFloat(){
	return  curand_uniform (&state);
}

__device__ float3 randomSpherePoint(){
	rndInit();
	//Init the parameters used to generate the random number
	float3 p = make_float3(curand_uniform (&state), curand_uniform (&state), curand_uniform (&state)) - make_float3(1, 1, 1) ;
	while (float3SquaredLength(p) >= 1.0) {
		p = make_float3(curand_uniform (&state), curand_uniform (&state), curand_uniform (&state)) - make_float3(1, 1, 1) ;
	}
	return p;
}

/**
 * Given a ray, create the blend of colors
 */
__device__ float3 color(ray &r, int sphereNumber, int* hittedMaterial, int d){

	hitRecord rec, tempRec;
	bool hitted = false;
	double closest = MAXFLOAT;

	for(int i = 0; i < sphereNumber; i++) {
		//Check if the ray hit the object
		//If there is already an hitted object it controls also if the new hitted object is closer than the previous
		if (spheres[i].hit(r, 0.01, closest, tempRec)) {
			hitted = true;
			closest = tempRec.t;
			rec = tempRec;
		}
	}

	//If the ray hitted something
	if (hitted && d < 2) {
		//Calculate the normal of the hitted objed
		float3 normal = (r.pointAtParam(rec.t) - rec.c) / rec.r;
		//Draw the normal
		*hittedMaterial = rec.objId;

		float3 target = r.pointAtParam(rec.t) + (0.5 * make_float3(normal.x + 1.0, normal.y + 1.0, normal.z + 1.0)) + randomSpherePoint();

		ray z = ray(r.pointAtParam(rec.t), target - r.pointAtParam(rec.t));
		return 0.5 * color(z , sphereNumber, &hittedMaterial[threadIdx.x], d + 1);//make_float3(normal.x + 1.0, normal.y + 1.0, normal.z + 1.0); //* color(z , sphereNumber, &hittedMaterial[threadIdx.x]);
	} else { //Draw the background
		float y = unitVector(r.direction()).y;
		float t = 0.5 * (y + 1.0);
		return (1.0 - t) * make_float3(1.0, 1.0, 1.0) + t * make_float3(0.5, 0.7, 1.0);
	}
}



__global__ void kernel(int* imageGpu, int sphereNumber){
	__shared__ int hittedMaterial[THRDSIZE];
	__shared__ bool anti;

	if (threadIdx.x == 0) anti = false;

	hittedMaterial[threadIdx.x] = 0;

	int offset = blockIdx.x * blockDim.x;
	int w = offset % WIDTH + threadIdx.x;
	int h = int(offset / WIDTH);

	rndInit();

	//u and v are used to translate a pixel coordinate on the scene
	float u =  float(w) / float(WIDTH);
	float v = float(h) / float(HEIGHT);

	//generate a ray that start from the origin and pass trough the center of a given pixel
	ray r(origin, lowerLeftCorner + (u * horizontal) + (v * vertical));
	//calculate the color that that ray sees
	float3 col = color(r, sphereNumber, &hittedMaterial[threadIdx.x], 0);

	__syncthreads();
	if (threadIdx.x == 0) {
		int firstHit = hittedMaterial[0];
		for (int i = 1; i < THRDSIZE; i++) {
			if(firstHit != hittedMaterial[i]) {
				anti = true;
			}
		}
	}
	__syncthreads();


	if (anti) { //Exec the antialiasing if necessary
		//Generate 9 ray equally spaced for each pixel
		for (int i = 0; i < 3; i ++) {
			for (int j = 0; j < 3; j ++){
				u =  (float(w) + 0.3 * i) / float(WIDTH);
				v = (float(h) + 0.3 * j) / float(HEIGHT);
				r = ray(origin, lowerLeftCorner + (u * horizontal) + (v * vertical));
				col = col + color(r, sphereNumber, &hittedMaterial[threadIdx.x], 0);
				col = col / 2;
			}
		}
		//in the end the value of color is the average color of the 9 rays
	}

	//Put the color seen by the ray in the memory address that correspond to the pixel
	imageGpu[addressConverter(h, w, 0)] = int(255.99 * sqrt(col.x));
	imageGpu[addressConverter(h, w, 1)] = int(255.99 * sqrt(col.y));
	imageGpu[addressConverter(h, w, 2)] = int(255.99 * sqrt(col.z));

}

__global__ void debug(){
	int offset = blockIdx.x * blockDim.x;
	int w = offset % WIDTH + threadIdx.x;
	int h = int(offset / WIDTH);

	rndInit();
	printf("0 - %f %f %f\n", randomSpherePoint().x, randomSpherePoint().y, randomSpherePoint().z);
	rndInit();
	printf("1 -  %f %f %f\n", randomSpherePoint().x, randomSpherePoint().y, randomSpherePoint().z);
	rndInit();
	printf("2 - %f %f %f\n", randomSpherePoint().x, randomSpherePoint().y, randomSpherePoint().z);
	rndInit();
	printf("3 - %f %f %f\n", randomSpherePoint().x, randomSpherePoint().y, randomSpherePoint().z);
}

int main ()
{
	cudaDeviceSetLimit(cudaLimitStackSize, 32768ULL);
	//Allocate a tridimensional vector that contains the image's data
	int *image;

	//cudaHostAlloc is used to allocate paged memory on the host device, in this way the position of the memory will
	//never change. It is necessary if we want to use asynchronous loading of the data in the gpu
	cudaHostAlloc((void**)&image, WIDTH * HEIGHT * BYTESPERPIXEL * sizeof( int ), cudaHostAllocDefault);
	gpuErrorCheck();

	memset((int*) image, 1, sizeof(int) * WIDTH * HEIGHT * BYTESPERPIXEL);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	//Initialize the data that will reside in the constant gpu memory in the
	float3 lowerLeftCornerCPU, horizontalCPU, verticalCPU, originCPU;
	cudaHostAlloc((void**) &lowerLeftCornerCPU, sizeof(float3), cudaHostAllocDefault);
	cudaHostAlloc((void**) &horizontalCPU, sizeof(float3), cudaHostAllocDefault);
	cudaHostAlloc((void**) &verticalCPU, sizeof(float3), cudaHostAllocDefault);
	cudaHostAlloc((void**) &originCPU, sizeof(float3), cudaHostAllocDefault);
	lowerLeftCornerCPU = make_float3(-2.0, -1.0, -1.0);
	horizontalCPU = make_float3(4.0, 0.0, 0.0);
	verticalCPU = make_float3(0.0, 2.0, 0.0);
	originCPU = make_float3(0.0, 0.0, 0.0);

	//Send the data to the constant memory
	cudaMemcpyToSymbolAsync(lowerLeftCorner, &lowerLeftCornerCPU, sizeof(float3), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(horizontal, &horizontalCPU, sizeof(float3), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(vertical, &verticalCPU, sizeof(float3), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(origin, &originCPU, sizeof(float3), 0, cudaMemcpyHostToDevice);
	gpuErrorCheck();

	//Allocate the image memory on the gpu
	int *imageGpu;
	cudaMalloc((void**) &imageGpu, sizeof(int) * WIDTH * HEIGHT * BYTESPERPIXEL);
	gpuErrorCheck();

	//Transfer the image to the gpu for the elaboration
	cudaMemcpyAsync(imageGpu, image, sizeof(int) * WIDTH * HEIGHT * BYTESPERPIXEL, cudaMemcpyHostToDevice, stream);
	gpuErrorCheck();

	sphere *spheresCPU[sn];

	spheresCPU[0] = new sphere(make_float3(0, 0, -1), 0.5, 1);
	spheresCPU[1] = new sphere(make_float3(10, -100.5, -1), 100, 2);

	for (int  i = 0; i <  sn; i++) {
		cudaMemcpyToSymbolAsync(spheres, spheresCPU[i], sizeof(sphere), sizeof(sphere) * i, cudaMemcpyHostToDevice);
	}
	gpuErrorCheck();

	kernel<<<BLKSIZE, THRDSIZE, 0, stream>>>(imageGpu, 2);
	//debug<<<2, 3>>>();
	gpuErrorCheck();

	//deallocate the constant data that now reside in the constant memory from the cpu
	//TODO deallocate i float3 che ora sono in memoria costante

	//Take back the image
	cudaMemcpyAsync(image, imageGpu,  sizeof(int) * WIDTH * HEIGHT * BYTESPERPIXEL, cudaMemcpyDeviceToHost, stream);
	gpuErrorCheck();

	//We have to be sure that all the data are back to the cpu
	cudaStreamSynchronize(stream);
	gpuErrorCheck(2);
	generate(WIDTH, HEIGHT, BYTESPERPIXEL, image);
	gpuErrorCheck(1);

	cudaFreeHost(image);
	gpuErrorCheck(0);

	printf("fine\n");
    return EXIT_SUCCESS;
}
