struct hitRecord {
	float t;
	float3 p;
	float3 normal;
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
			if ((b * b - a*c) > 0){
				//TODO riscrivi questa parte per renderla più estetica
				float temp = (-b - sqrt(b * b -  a * c)) / a;
				if (temp < tMax && temp > tMin) {
					rec.t = temp;
					rec.p = r.pointAtParam(rec.t);
					rec.normal = (rec.p - center) / radius;
					return true;
				}
				temp = (-b - sqrt(b * b -  a * c)) / a;
				if (temp < tMax && temp > tMin) {
					rec.t = temp;
					rec.p = r.pointAtParam(rec.t);
					rec.normal = (rec.p - center) / radius;
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
			for(int i = 0; i < listSize; i++) {
				if (list[i]->hit(r, t_min, closest, tempRec)) {
					hitted = true;
					closest = tempRec.t;
					rec = tempRec;
				}
			}
			return hitted;
	}
};
