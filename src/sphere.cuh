#include "texture.h"

enum materials {
	MATTE = 1,
	METAL = 2,
	GLASS = 3,
};

//When there is an hit we store the parameters of the hitted object to do the calculation
//only on the closest hitted object
struct hitRecord {
	float t; //Position where the sphere is hitten
	float3 c; //center of the sphere
	float r; //radius of hte sphere

	//Coordinates where the ray hit the object
	float x;
	float y;

	int objId;
	float3 color;
	materials material;

	/**
	 * When a ray hit a dielectric material (like glass) the ray is either reflected or refracted,
	 * in other words the ray bounce on the sourface (reflected ray) or penetrate the surface
	 * (refracted ray.). The ratio of reflection rays is given by a constant for each material
	 * for instance water has a refraction index of 1.33, diamonds 2.44
	 */
	float refractionIndex;

	int textureId;
};



/**
 * Determine the direction of reflected rays
 */
__device__ float3 reflect (const float3 &v, const float3 &n) {
	return v - 2*dot(v, n) * n;
}

/**
 * Determine the direction of refracted rays
 */
__device__ bool refract(const float3 v, const float3 normal, float niOverNt, float3 &refracted){
	float3 uv = unitVector(v);
	float dt = dot(uv, normal);
	float discriminant = 1.0 - niOverNt * niOverNt * (1 - dt*dt);
	if (discriminant > 0) {
		refracted = niOverNt * (uv - dt * normal) - sqrt(discriminant) * normal;
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

		int textureId;

		__device__ __host__ sphere() {};
		__device__ __host__ sphere(float3 cen,  float rad, int i, float3 col, materials m, float ref = 1, int texId = 0) {
			center = cen;
			radius = rad;
			id = i;
			color = col;
			material = m;
			refractionIndex = ref;
			textureId = texId;
		};

		/**
		 * Get the position hitted by the ray
		 */
		__device__ static void getSphereXY(const float3 &p, float &x, float &y) {

			float theta = acos(-p.y);
			float phi = atan2f(-p.z, p.x ) + PI;

			x = phi / (2 * PI);
			y = theta / PI;
		}

		/**
		 * Check if a given ray hits this object
		 */
		__device__ bool hit(const ray &r, float tMin, float tMax, hitRecord &rec) const {
			float3 oc = r.origin() - center;
			float a = dot(r.direction(), r.direction());
			float b = dot(oc, r.direction());
			float c = dot(oc, oc) - radius*radius;

			//The sphere may be hitted never, once or two time
			//If this is more than 0 it means that eventually the ray hit the sphere
			if ((b * b - a*c) > 0){
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
					rec.textureId = textureId;
					getSphereXY((r.pointAtParam(temp) - center) / radius, rec.x, rec.y);
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
					rec.textureId = textureId;
					getSphereXY((r.pointAtParam(temp) - center) / radius, rec.x, rec.y);
					return true;
				}
			}
			return false;
		}
};
