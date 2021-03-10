#pragma once
#include <errno.h>

#define addressConverterTexture(H, W, P, MAXW) (((W) * (3)) + ((H) * (MAXW) * (3)) + (P))

//TODO create an enum to distinguish default textures
//TODO create the grid texture

class myTexture {
	public:
		int height;
		int width;
		int *img;

		__device__ __host__ myTexture() {};
		myTexture (char* name) {

			printf("Start loading texture %s\n", name);

			char fullPath[256] = "";
			fullPath[0] = '\0';
			strcat(fullPath, "./textures/");
			strcat(fullPath, name);
			printf("%s\n", fullPath);

			FILE *ppm = fopen(fullPath, "r");

			//
			if (ppm == NULL) {
				perror("Error opening texture.\n");
			}


			char buffer[10000];

			//Read first line and check if it is "P3"
			 if(fgets (buffer, 100, ppm) != NULL){
				if (strcmp(buffer, "P3")) printf("The convertion type of the ppm image of texture %s is right\n", name);
				else printf("The %s texture can't be read, only ppm P3 file are accepted \n", name);
			 }

			 //Read sizes of the image
			 if(fgets (buffer, 100, ppm) != NULL){
				 width= atoi(strtok (buffer, " "));
				 height = atoi(strtok (NULL, " "));
			 }

			 //Ignore the maximum value
			 if(fgets (buffer, 100, ppm) == NULL){
				 printf("Unexprected value while loading %s texture\n", name);
			 }

			 //Allocate enough space to store the image data.
			 cudaHostAlloc((void**)&img, width * height * BYTESPERPIXEL * sizeof( int), cudaHostAllocDefault);
			 //img = (int*) malloc(width * height * BYTESPERPIXEL * sizeof( int));

			 //Read all the lines of the file
			 int x = 0;
			 int pixel = 0;
			 int y = height - 1;


			 char *value;
			 while (fgets (buffer, 10000, ppm) != NULL) { //A line rs read till there is something to read
				 value = strtok (buffer, " "); //strtok return a single word of the line each time it is called
				 while (value != NULL) { //Thill there is something in the buffer
					 img[addressConverterTexture(y, x, pixel, width)] =  atoi(value); //The value of a pixel is set based on the value read
					 value = strtok (NULL, " ");
					 if(value == NULL){ //If in the buffer there is nothin, read the next line
						 break;
					 }
					 pixel ++; //Update the pixel counter
					 if (pixel > 2) {
						 pixel = 0;
						 x ++;
						 if (x > width) {
							 x = 0;
							 y--;
						 }
					 }


				 }
			 }
			 printf("Texture %s correctly loaded\n", name);
			 fclose(ppm);
		}

		__device__ __host__ int *getImg(){ return img; }

		__device__ __host__ int getWidth(){ return width; }

		__device__ __host__ int getHeight(){	return height; }
};
