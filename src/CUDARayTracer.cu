#include <stdio.h>
#include "imageGenerator.h"
#include "constants.h"
#include "float3.h"
#include "ray.h"
#include "hitable.h"
#include "random.cuh"



//TODO write some comments

//--Optional
//TODO generate numberm more randomly
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
const int sphereNumber = 10;

/**
 * Check for gpu errors
 */
void gpuErrorCheck(int i = 0){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		    printf("Error %d: %s\n", i, cudaGetErrorString(err));
}

/**
 * Calculate the color iteratively
 */
__device__ float3 color(ray &r, int sphereNumber, int* hittedMaterial, int maxIteration){

	ray currentRay;
	currentRay.A = r.origin();
	currentRay.B = r.direction();
	int iteration = 0;
	float3 colorMultiplier = make_float3(1, 1, 1);

	while(true) {
		hitRecord rec, tempRec;
		bool hitted = false;
		double closest = MAXFLOAT;

		//For each object check if the ray hit that object
		for(int i = 0; i < sphereNumber; i++) {
			//Check if the ray hit the object
			//If there is already an hitted object it controls also if the new hitted object is closer than the previous
			if (spheres[i].hit(currentRay, 0.001, closest, tempRec)) {
				hitted = true;
				closest = tempRec.t;
				rec = tempRec;
			}
		}

		//If the ray hitted something
		if (hitted && iteration < maxIteration) {
			iteration ++;
			//Register the hitted object
			*hittedMaterial = rec.objId;

			//Calculate the normal of the hitted objed
			float3 normal = (currentRay.pointAtParam(rec.t) - rec.c) / rec.r;

			if (rec.material == MATTE) {
				//TODO comment that
				float3 target = currentRay.pointAtParam(rec.t) + normal + randomSpherePoint();

				currentRay = ray(currentRay.pointAtParam(rec.t), target - currentRay.pointAtParam(rec.t));

				//After each iteration a percent of the light is absorbed
				colorMultiplier = rec.color * colorMultiplier;
			} else if (rec.material == METAL) {
				float3 reflected = reflect(unitVector(currentRay.direction()) , normal);

				currentRay= ray(currentRay.pointAtParam(rec.t), reflected);

				colorMultiplier = rec.color * colorMultiplier;
				iteration = maxIteration - 1;
			} else if (rec.material == GLASS) {

				float3 outwardNormal;
				float3 reflected = reflect(currentRay.direction(), normal);
				float3 attenuation = make_float3(1, 1, 1);
				float niOverNt;

				if (dot(currentRay.direction(), normal) > 0) { //the rau face outside ??
					outwardNormal = -1 * normal;
					niOverNt = rec.refractionIndex;
				} else { //The ray face inside
					outwardNormal = normal;
					niOverNt = 1.0 / rec.refractionIndex;
				}

				float3 refracted;
				if(!refract(currentRay.direction(), outwardNormal, niOverNt, refracted))
					iteration = maxIteration;

				currentRay = ray(currentRay.pointAtParam(rec.t), refracted);
				colorMultiplier = attenuation * colorMultiplier;
			}else if (rec.material == LIGHT){
				return rec.color;
			}

		} else { //Draw the background
			float y = unitVector(r.direction()).y;
			float t = 0.5 * (y + 1.0);
			colorMultiplier = colorMultiplier * ((1.0 - t) * make_float3(1.0, 1.0, 1.0) + t * make_float3(0.5, 0.7, 1.0));
			return colorMultiplier;
	}
	}
}



__global__ void kernel(int* imageGpu, int sphereNumber, int maxIteration){
	/**
	 * This array of hitted material keep track of the materials hitted by the thread
	 * in the same block.
	 */
	__shared__ int hittedMaterial[THRDSIZE];


	// This variable control if a block has to execute the raytracing
	__shared__ bool performRaytracing;

	//Only one thread per block set this variable to false
	if (threadIdx.x == 0) performRaytracing = false;

	//All the array of hitted objects is iniitalized to the same value
	hittedMaterial[threadIdx.x] = 0;

	//The memory location where each thread has to operate is initialized
	int offset = blockIdx.x * blockDim.x;

	//width and height coordinates are calculated
	int w = offset % WIDTH + threadIdx.x;
	int h = int(offset / WIDTH);

	//u and v are used to translate a pixel coordinate on the scene
	float u =  float(w) / float(WIDTH);
	float v = float(h) / float(HEIGHT);

	//generate a ray that start from the origin and pass trough the center of a given pixel
	ray r(origin, lowerLeftCorner + (u * horizontal) + (v * vertical));
	//calculate the color that that ray sees
	float3 col = color(r, sphereNumber, &hittedMaterial[threadIdx.x], maxIteration);

	//One thread per block checks if all the other threads hitted the same object
	__syncthreads();
	if (threadIdx.x == 0) {
		int firstHit = hittedMaterial[0];
		for (int i = 1; i < THRDSIZE; i++) {
			if(firstHit != hittedMaterial[i]) {
				performRaytracing = true;
			}
		}
	}
	__syncthreads();



	if (performRaytracing) { //Exec the antialiasing if necessary
		//Generate 100 ray equally spaced for each pixel
		for (int i = -5; i < 5; i ++) {
			for (int j = -5; j < 5; j ++){
				u =  (float(w) + 0.1 * i) / float(WIDTH);
				v = (float(h) + 0.1 * j) / float(HEIGHT);
				r = ray(origin, lowerLeftCorner + (u * horizontal) + (v * vertical));
				col = col + color(r, sphereNumber, &hittedMaterial[threadIdx.x], maxIteration);
				col = col / 2;
			}
		}
		//col = make_float3(1, 0, 0);
		//in the end the value of color is the average color of the 100 rays
	}

	//Put the color seen by the ray in the memory address that correspond to the pixel
	imageGpu[addressConverter(h, w, 0)] = int(255.99 * col.x);
	imageGpu[addressConverter(h, w, 1)] = int(255.99 * col.y);
	imageGpu[addressConverter(h, w, 2)] = int(255.99 * col.z);

}

int main ()
{
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

	sphere *spheresCPU[sphereNumber];

	spheresCPU[0] = new sphere(make_float3(0, -100.5, -1), 100, 0, make_float3(0.7, 0.7, 0.7), MATTE);
	spheresCPU[1] = new sphere(make_float3(-1.5, -0.2, -1.3), 0.4, 1, make_float3(0.7, 0.3, 0.3), MATTE);
	spheresCPU[2] = new sphere(make_float3(-0.5, -0.2, -1.3), 0.4, 2, make_float3(0.9, 0.4, 0.4), GLASS, 1.5);
	spheresCPU[3] = new sphere(make_float3(0.5, -0.2, -1.3), 0.4, 3, make_float3(0.8, 0.6, 0.2), LIGHT);
	spheresCPU[4] = new sphere(make_float3(1.5, -0.2, -1.3), 0.4, 4, make_float3(0.8, 0.8, 0.8), METAL);

	spheresCPU[5] = new sphere(make_float3(1,  -0.3, -0.9), 0.2, 5, make_float3(0.9, 0.9, 0.9), MATTE);
	spheresCPU[6] = new sphere(make_float3(-1, -0.3, -0.9), 0.2, 6, make_float3(0.3, 0.3, 0.9), MATTE);
	spheresCPU[7] = new sphere(make_float3(-0.0, -0.3, -0.9), 0.2, 7, make_float3(0.9, 0.4, 0.4), GLASS, 2.44);
	spheresCPU[8] = new sphere(make_float3(-0.5, -0.3, -0.9), 0.2, 8, make_float3(0.1, 0.1, 0.7), LIGHT);
	spheresCPU[9] = new sphere(make_float3(0.5, -0.3, -0.9), 0.2, 9, make_float3(0.2, 0.2, 0.8), METAL);

	for (int  i = 0; i <  sphereNumber; i++) {
		cudaMemcpyToSymbolAsync(spheres, spheresCPU[i], sizeof(sphere), sizeof(sphere) * i, cudaMemcpyHostToDevice);
	}
	gpuErrorCheck();

	kernel<<<BLKSIZE, THRDSIZE, 0, stream>>>(imageGpu, sphereNumber, 10);

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
