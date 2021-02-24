
class ray{
public:
	float3 A; //origin of the ray
	float3 B; //direction of the ray

	__device__ ray() {}
	__device__ ray(const float3 &a, const float3 &b) {
		A = a;
		B = b;
	}

	__device__ float3 origin() const {return A;}
	__device__ float3 direction() const {return B;}
	//This equation gives the position of the ray when t varies
	__device__ float3 pointAtParam(float t) const {return A + (t *B);}

};
