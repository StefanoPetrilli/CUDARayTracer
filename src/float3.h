#include <math.h>

__device__  float float3Length(const float3 &a){ return sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }

__device__ float float3SquaredLength(const float3 &a){ return a.x * a.x + a.y * a.y + a.z * a.z; }

__device__ float3 operator+ (const float3 &a, const float3 &b){ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

__device__ float3 operator* (float t, const float3 &a){ return make_float3(a.x  * t, a.y *t, a.z *t); }

__device__ float3 operator* (const float3  &b, const float3 &a){ return make_float3(a.x  * b.x, a.y * b.y, a.z  * b.z); }

__device__ float3 operator/ (const float3 &a, const float t){ return make_float3(a.x  / t, a.y  / t, a.z  / t); }

__device__ float3 operator/ (const float3 &a, const float3 &b){ return make_float3(a.x  / b.x, a.y  / b.y, a.z  / b.z); }

__device__ float3 operator- (const float3 &a, const float3 &b){ 	return make_float3(a.x  - b.x, a.y  - b.y, a.z  - b.z); }

__device__ float dot(const float3 &u, const float3 &v) { return (u.x * v.x) + (u.y * v.y) + (u.z * v.z); }

//Return a vector of length 1.
__device__ float3 unitVector(float3 v){ return v / float3Length(v); }

__device__ void operator+= (float3 &a, const float3 &b){
	a.x  += b.x;
	a.y  += b.y;
	a.z  += b.z;
}

__device__ void operator/= (float3 &a, const int b){
	a.x /= b;
	a.y /= b;
	a.z /= b;
}

__device__ void operator*= (float3 &a, const float3 &b){
	a.x  *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
