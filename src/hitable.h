enum materials {
	MATTE = 1,
	METAL = 2,
	GLASS = 3,
	LIGHT = 4
};

//When there is an hit we store the parameters of the hitted object to do the calculation
//only on the closest hitted object
struct hitRecord {
	float t;
	float3 c;
	float r;
	int objId;
	float3 color;
	materials material;
	float refractionIndex;
};

__device__ float3 reflect (const float3 &v, const float3 &n) {
	return v - 2*dot(v, n) * n;
}

__device__ bool refract(const float3 v, const float3 normal, float niOverNt, float3 &refracted){
	float3 uv = unitVector(v);
	float dt = dot(uv, normal);
	float discriminant = 1.0 - niOverNt * niOverNt * (1 - dt*dt);
	if (discriminant > 0) {
		refracted = niOverNt * (uv - normal*dt) - normal * sqrt(discriminant);
		return true;
	}
	return false;
}

class sphere {
	public:
		float3 center;
		float radius;
		int id;
		float3 color;
		materials material;
		float refractionIndex;

		__device__ __host__ sphere() {}
		__device__ __host__ sphere(float3 cen,  float rad, int i, float3 col, materials m, float ref = 1) {
			center = cen;
			radius = rad;
			id = i;
			color = col;
			material = m;
			refractionIndex = ref;
		};

		__device__ bool hit(const ray &r, float tMin, float tMax, hitRecord &rec) const {
			float3 oc = r.origin() - center;
			float a = dot(r.direction(), r.direction());
			float b = dot(oc, r.direction());
			float c = dot(oc, oc) - radius*radius;

			//The sphere may be hitted never, once or two time
			//If this is more than 0 it means that eventually the ray hit the sphere
			if ((b * b - a*c) > 0){
				//TODO riscrivi questa parte per renderla più bella
				//Calculate the first hitting point
				float temp = (-b - sqrt(b * b -  a * c)) / a;
				//If the hitting point is in front of the camera we register that hit
				if (temp < tMax && temp > tMin) {
					//If it is hitted save the position and the parameters of the sphere that is hitted
					rec.t = temp;
					rec.c = center;
					rec.r = radius;
					rec.objId = id;
					rec.color = color;
					rec.material = material;
					rec.refractionIndex = refractionIndex;
					return true;
				}
				//Do the same with the second hitting point
				temp = (-b - sqrt(b * b -  a * c)) / a;
				if (temp < tMax && temp > tMin) {
					rec.t = temp;
					rec.c = center;
					rec.r = radius;
					rec.objId = id;
					rec.color = color;
					rec.material = material;
					rec.refractionIndex = refractionIndex;
					return true;
				}
			}
			return false;
		}
};
