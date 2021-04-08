# CUDARayTracer
![Alt Text](https://s4.gifyu.com/images/Untitled8772bc18b6640d85.gif)

Raytracers are part of embarrassingly parallel problems. The problems of this kind, require little or no effort to be divided into an arbitrary number of parallel tasks.
In fact, in my implementation, a small group of pixels are computed independently from all the other pixels.
In my opinion, this class of problems are very effective to practice CUDA because the parallelization itself is easy and one can focus on other trivial aspects of this API.

This code has been written as the final project of the course Computer Graphics at Sapienza University. The raytracer itself is based on "Raytracing in one weekend", "Raytracing the next week" and the slide of the course.
