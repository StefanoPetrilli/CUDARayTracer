#include <stdio.h>
#include "imageGenerator.h"
#include "constants.h"
#include "float3.h"
#include "ray.h"
#include "sphere.cuh"
#include "random.cuh"
#include "texture.h"
#include <stdio.h>

//TODO Features that could be implemented
// build scenes with a configuration file
// order the materials rendering to exploit warps of GPU

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

/**
 *The nvidia's GPUs have texture memory. This memory is cached in a way that
 *optimizes spatial locality, so threads that read addresses that are close together
 *will achieve better performance..
 */
texture<int> textureOne;
texture<int> textureTwo;
texture<int> textureThree;

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
 * Calculate the color that the ray sees iteratively
 */
__device__ float3 color(ray &r, int sphereNumber, int* hittedMaterial, int maxIteration){

	ray currentRay;
	currentRay.A = r.origin();
	currentRay.B = r.direction();
	int iteration = 0;
	float3 initialLight = make_float3(1, 1, 1);

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

		//If the ray hitted something and we are not in the last iteration
		if (hitted && iteration < maxIteration) {

			//Register the hitted object
			if(iteration == 0) *hittedMaterial = rec.objId;
			iteration ++;

			//Calculate the normal of the hitted objed
			float3 normal = (currentRay.pointAtParam(rec.t) - rec.c) / rec.r;

			if (rec.material == MATTE) {
				//the new ray bounce in the direction of the normal plus a random deviation
				float3 target = currentRay.pointAtParam(rec.t) + normal + randomSpherePoint();
				//the new ray is created
				currentRay = ray(currentRay.pointAtParam(rec.t), target - currentRay.pointAtParam(rec.t));

				//After each iteration a percent of the light is absorbed
				if(rec.textureId == 0) 	initialLight *= rec.color;
				else {

					//Load the correct dimension of the texture
					int textureX, textureY;
					switch (rec.textureId) {
					case 1:
						textureX = tex1Dfetch(textureOne, 0) ;
						textureY = tex1Dfetch(textureOne, 1);
						break;
					case 2:
						textureX = tex1Dfetch(textureTwo, 0) ;
						textureY = tex1Dfetch(textureTwo, 1) ;
						break;
					default:
						break;
					}

					//Wrap the image with several repetition of the texture based on the ray of the sphere
					float textureXRep = rec.x * rec.r;
					while (textureXRep > 1.0) textureXRep -= 1.0;

					float textureYRep = rec.y * rec.r;
					while (textureYRep > 1.0) textureYRep -= 1.0;

					//Translate the hitpoint texture coordinates
					int pixelCoordX = textureX * textureXRep;
					int pixelCoordY = textureY * (1.0 - textureYRep);

					int r, g, b;
					switch (rec.textureId) {
						case 1:
							r = tex1Dfetch(textureOne, addressConverterTexture(pixelCoordY, pixelCoordX, 0, textureX));
							g = tex1Dfetch(textureOne, addressConverterTexture(pixelCoordY, pixelCoordX, 1, textureX));
							b = tex1Dfetch(textureOne, addressConverterTexture(pixelCoordY, pixelCoordX, 2, textureX));
							break;
						case 2:
							r = tex1Dfetch(textureTwo, addressConverterTexture(pixelCoordY, pixelCoordX, 0, textureX));
							g = tex1Dfetch(textureTwo, addressConverterTexture(pixelCoordY, pixelCoordX, 1, textureX));
							b = tex1Dfetch(textureTwo, addressConverterTexture(pixelCoordY, pixelCoordX, 2, textureX));
							break;
						default:
						break;
					}

					initialLight = make_float3(r / 256.0, g / 256.0, b / 256.0) * initialLight;

				}
			} else if (rec.material == METAL) {

				//For sharp metal material the light is just reflected on the normal
				float3 reflected = reflect(unitVector(currentRay.direction()) , normal);
				currentRay= ray(currentRay.pointAtParam(rec.t), reflected);

				initialLight = rec.color * initialLight;

				//We can stop after checking the color hitted by the reflected ray
				iteration = maxIteration - 1;

			} else if (rec.material == GLASS) {

				float3 outwardNormal;
				float3 reflected = reflect(currentRay.direction(), normal);
				float3 attenuation = make_float3(1, 1, 1);
				float niOverNt;

				if (dot(currentRay.direction(), normal) > 0) { //the ray face outside
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
				initialLight = attenuation * initialLight;
			}

		} else { //Draw a gradient that goes form sky blue on top to white on the bottom
			float y = unitVector(r.direction()).y;
			float t = 0.5 * (y + 1.0);
			initialLight = initialLight * ((1.0 - t) * make_float3(1.0, 1.0, 1.0) + t * make_float3(0.5, 0.7, 1.0));
			return initialLight;
	}
	}
}


/**
 * Code executed by each thread of the GPU
 */
__global__ void kernel(float* imageGpu, int sphereNumber, int maxIteration){

	// This variable control if a block has to execute the raytracing
	__shared__ bool performAntialiasing;

	/**
	 * This array of hitted material keep track of the materials hitted by the thread
	 * in the same block.
	 */
	__shared__ int hittedMaterial[THRDSIZE];


	//Only one thread per block set this variable to false
	if (threadIdx.x == 0) performAntialiasing = false;
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
	__syncthreads(); //This is a barrier, all the threads stops here
	if (threadIdx.x == 0) { //Only one thread for each block execute this block of code
		int firstHit = hittedMaterial[0];
		for (int i = 1; i < THRDSIZE; i++) {
			if(firstHit != hittedMaterial[i]) {
				performAntialiasing = true;
				break;
			}
		}
	}
	__syncthreads();

	if (performAntialiasing) { //Exec the antialiasing if necessary
		//Generate 16 ray equally spaced for each pixel
		for (int i = -2; i < 2; i ++) {
			for (int j = -2; j < 2; j ++){
				u =  (float(w) + 0.24 * i) / float(WIDTH);
				v = (float(h) + 0.24 * j) / float(HEIGHT);
				r = ray(origin, lowerLeftCorner + (u * horizontal) + (v * vertical));
				col +=  color(r, sphereNumber, &hittedMaterial[threadIdx.x], maxIteration);
				col /= 2;
			}
		}
		//in the end the value of color is the average color of the 16 rays
		if (ANTIALIASINGDEBUG) col /= ITERATIONS;
	}

	//Put the color seen by the ray in the memory address that correspond to the pixel
	imageGpu[addressConverter(h, w, 0)] += 255.99 * col.x ;
	imageGpu[addressConverter(h, w, 1)] += 255.99 * col.y;
	imageGpu[addressConverter(h, w, 2)] += 255.99 * col.z;
}

int main ()
{
	//Events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Allocate a tridimensional vector that contains the image's data
	float *image;

	//cudaHostAlloc is used to allocate paged memory on the host device, in this way the position of the memory will
	//never change. It is necessary if we want to use asynchronous loading of the data in the gpu
	cudaHostAlloc((void**)&image, WIDTH * HEIGHT * BYTESPERPIXEL * sizeof( float ), cudaHostAllocDefault);
	gpuErrorCheck();

	memset((float*) image, 0, sizeof(float) * WIDTH * HEIGHT * BYTESPERPIXEL);

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
	float *imageGpu;
	cudaMalloc((void**) &imageGpu, sizeof(float) * WIDTH * HEIGHT * BYTESPERPIXEL);
	gpuErrorCheck();

	//Transfer the image to the gpu for the elaboration
	cudaMemcpyAsync(imageGpu, image, sizeof(float) * WIDTH * HEIGHT * BYTESPERPIXEL, cudaMemcpyHostToDevice, stream);
	gpuErrorCheck();

	//Load textures
	myTexture m1 = myTexture(TEXONENAME);
	myTexture m2 = myTexture(TEXTWONAME);
	myTexture m3 = myTexture(TEXTWONAME);

	int *tex1 = m1.getImg();
	int *tex2 = m2.getImg();
	int *tex3 = m3.getImg();

	tex1[0] = m1.getWidth();
	tex1[1] = m1.getHeight();

	tex2[0] = m2.getWidth();
	tex2[1] = m2.getHeight();

	tex3[0] = m3.getWidth();
	tex3[1] = m3.getHeight();

	cudaMalloc((void**) &textureOne, sizeof(int) * m1.getWidth() * m1.getHeight() * BYTESPERPIXEL);
	cudaMalloc((void**) &textureTwo, sizeof(int) * m2.getWidth() * m2.getHeight() * BYTESPERPIXEL);
	cudaMalloc((void**) &textureThree, sizeof(int) * m3.getWidth() * m3.getHeight() * BYTESPERPIXEL);
	gpuErrorCheck();

	cudaBindTexture(NULL, textureOne, tex1, sizeof(int) * m1.getWidth() * m1.getHeight() * BYTESPERPIXEL);
	cudaBindTexture(NULL, textureTwo, tex2, sizeof(int) * m2.getWidth() * m2.getHeight() * BYTESPERPIXEL);
	cudaBindTexture(NULL, textureThree, tex2, sizeof(int) * m3.getWidth() * m3.getHeight() * BYTESPERPIXEL);
	gpuErrorCheck();

	//////////////////////////////////////////////////
	// CREATE THE SCENE TO BE RENDERED //
	/////////////////////////////////////////////////

	sphere *spheresCPU[sphereNumber];

	spheresCPU[0] = new sphere(make_float3(0, -100.5, -1), 100, 0, make_float3(0.7, 0.7, 0.7), MATTE, 1, 2);
	spheresCPU[1] = new sphere(make_float3(-1.5, -0.2, -1.3), 0.4, 1, make_float3(0.9, 0.9, 0.9), MATTE);
	spheresCPU[2] = new sphere(make_float3(-0.5, -0.2, -1.3), 0.4, 2, make_float3(0.9, 0.4, 0.4), GLASS, 2.44);
	spheresCPU[3] = new sphere(make_float3(0.5, -0.2, -1.3), 0.4, 3, make_float3(0.8, 0.6, 0.2), MATTE, 1, 1);
	spheresCPU[4] = new sphere(make_float3(1.5, -0.2, -1.3), 0.4, 4, make_float3(0.8, 0.8, 0.8), METAL);

	spheresCPU[5] = new sphere(make_float3(1,  -0.3, -0.9), 0.2, 5, make_float3(0.89, 0.12, 0.39), MATTE);
	spheresCPU[6] = new sphere(make_float3(-1, -0.3, -0.9), 0.2, 6, make_float3(0.99, 0.43, 0.25), MATTE);
	spheresCPU[7] = new sphere(make_float3(-0.5, -0.3, -0.9), 0.2, 7, make_float3(1, 1, 1), METAL);
	spheresCPU[8] = new sphere(make_float3(-0.0, -0.3, -0.9), 0.2, 8, make_float3(0.11, 0.51, 0.5), GLASS, 1.1);
	spheresCPU[9] = new sphere(make_float3(0.5, -0.3, -0.9), 0.2, 9, make_float3(0.2, 0.2, 0.8), METAL);

	//Load the objects in the memory
	for (int  i = 0; i <  sphereNumber; i++) {
		cudaMemcpyToSymbolAsync(spheres, spheresCPU[i], sizeof(sphere), sizeof(sphere) * i, cudaMemcpyHostToDevice);
	}
	gpuErrorCheck();

	//Do the actual computation for  each iteration
	char buffer[4];
	float milliseconds;
	for (int i = 0; i < ITERATIONS; i++){
		cudaEventRecord(start);
		kernel<<<BLKSIZE, THRDSIZE, 0, stream>>>(imageGpu, sphereNumber, MAXRECURSIONDEPTH );
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Iteration %d elapsed in %f milliseconds\n", i, milliseconds);
		if (SAVEALLITERATIONS) {
			cudaMemcpyAsync(image, imageGpu,  sizeof(int) * WIDTH * HEIGHT * BYTESPERPIXEL, cudaMemcpyDeviceToHost, stream);
			cudaStreamSynchronize(stream);
			char dir[80] = "output";
			snprintf(buffer, sizeof(buffer), "%d", i);
			strcat(dir, buffer);
			strcat(dir, ".ppm");
			generate(WIDTH, HEIGHT, BYTESPERPIXEL, image, i, dir);
		}
	}
	gpuErrorCheck(1);

	if(!SAVEALLITERATIONS) {
		//Take back the image
		cudaMemcpyAsync(image, imageGpu,  sizeof(int) * WIDTH * HEIGHT * BYTESPERPIXEL, cudaMemcpyDeviceToHost, stream);
		gpuErrorCheck();

		//We have to be sure that all the data are back to the cpu
		cudaStreamSynchronize(stream);
		gpuErrorCheck();
		generate(WIDTH, HEIGHT, BYTESPERPIXEL, image, ITERATIONS);
		gpuErrorCheck();
	}

	//Deallocate memory
	cudaFreeHost(image);
	cudaFreeHost(&lowerLeftCornerCPU);
	cudaFreeHost(&horizontalCPU);
	cudaFreeHost(&verticalCPU);
	cudaFreeHost(&originCPU);
	cudaFreeHost(&spheresCPU);

	printf("Successfully terminated \n");
	return EXIT_SUCCESS;
}
