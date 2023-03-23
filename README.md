# CUDARayTracer

![1617905515202](https://user-images.githubusercontent.com/8702339/227385787-a111f5ac-ce6d-474f-9509-ef91f8f1c35e.jpeg)

Raytracers are part of embarrassingly parallel problems. The problems of this kind, require little or no effort to be divided into an arbitrary number of parallel tasks.
In fact, in my implementation, a small group of pixels are computed independently from all the other pixels.
In my opinion, this class of problems are very effective to practice CUDA because the parallelization itself is easy and one can focus on other trivial aspects of this API.

This code has been written as the final project of the course Computer Graphics at Sapienza University. The raytracer itself is based on "Raytracing in one weekend", "Raytracing the next week" and the slide of the course.
