#include <curand.h>
#include <curand_kernel.h>

__device__ curandState_t state;

/**
 *  Generate random number using the cuda library CUrand
 */
__device__ int pw = 0;
__device__ void rndInit(){

	pw = pw + threadIdx.x  + blockIdx.x * blockDim.x;
	//Init the parameters used to generate the random number
	curand_init(pw, 0, 0, &state);
}

__device__ float getRndFloat(){
	float p = curand_uniform (&state);
	//printf("%f \n", p);
	return  p;
}

__device__ float3 randomSpherePoint(){
	rndInit();
	//Init the parameters used to generate the random number
	float3 p = 2.0 * make_float3(curand_uniform (&state), curand_uniform (&state), curand_uniform (&state)) - make_float3(1, 1, 1) ;
	while (float3SquaredLength(p) >= 1.0) {
		p = 2.0 * make_float3(curand_uniform (&state), curand_uniform (&state), curand_uniform (&state)) - make_float3(1, 1, 1) ;
	}
	return p;
}
