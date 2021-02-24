//When there is an hit we store the parameters of the hitted object to do the calculation
//only on the closest hitted object
struct hitRecord {
	float t;
	float3 c;
	float r;
};

class hitable {
	public:
		__device__ virtual bool hit(const ray &r, float t_min, float t_max, hitRecord &rec) const = 0;
};

class sphere: public hitable {
	public:
		float3 center;
		float radius;

		__device__ sphere() {}
		__device__ sphere(float3 cen,  float rad) {
			center = cen;
			radius = rad;
		};

		__device__ bool hit(const ray &r, float tMin, float tMax, hitRecord &rec) const {
			float3 oc = r.origin() - center;
			float a = dot(r.direction(), r.direction());
			float b = dot(oc, r.direction());
			float c = dot(oc, oc) - radius*radius;

			//The sphere may be hitted never, once or two time
			//If this is more than 0 it means that eventually the ray hit the sphere
			if ((b * b - a*c) > 0){
				//TODO riscrivi questa parte per renderla più estetica
				//Calculate the first hitting point
				float temp = (-b - sqrt(b * b -  a * c)) / a;
				//If the hitting point is in front of the camera we register that hit
				if (temp < tMax && temp > tMin) {
					//If it is hitted save the position and the parameters of the sphere that is hitted
					rec.t = temp;
					rec.c = center;
					rec.r = radius;
					return true;
				}
				//Do the same with the second hitting point
				temp = (-b - sqrt(b * b -  a * c)) / a;
				if (temp < tMax && temp > tMin) {
					rec.t = temp;
					rec.c = center;
					rec.r = radius;
					return true;
				}
			}
			return false;
		}
};

class hitableList: public hitable {
public:

	hitable **list;
	int listSize;

	__device__ hitableList() {}
	__device__ hitableList(hitable **l, int n) {list = l; listSize = n;}

	__device__ virtual bool hit(const ray &r, float t_min, float t_max, hitRecord &rec) const {
			hitRecord tempRec;
			bool hitted = false;
			double closest = t_max;
			//Foreach hitable object
			for(int i = 0; i < listSize; i++) {
				//Check if the ray hit the object
				//If there is already an hitted object it controls also if the new hitted object is closer than the previous
				if (list[i]->hit(r, t_min, closest, tempRec)) {
					hitted = true;
					closest = tempRec.t;
					rec = tempRec;
				}
			}
			return hitted;
	}
};
