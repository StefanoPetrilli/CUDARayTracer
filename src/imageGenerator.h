#include <stdio.h>
#include "constants.h"

/**
 * Function that generates the image
 */
void generate(int width, int height, int bytesPerPixel, float* image, int iterations, char* imageName = "output.ppm" ){

	char dir[80] = "./images/";
	strcat(dir, imageName);
	printf("%s\n", dir);
	FILE *ppm = fopen(dir, "wb");

	//Write the image's header
	fprintf(ppm, "P3\n");
	fprintf(ppm, "%d %d\n255\n", width, height);
	//Append to the file all the pixels
	for (int y = height - 1; y > 0; y--){
			for (int x = 0; x < width; x++) {

				fprintf(ppm, "%d %d %d\n", int(image[addressConverter(y, x, 0)] / (iterations + 1)),
					int(image[addressConverter(y, x, 1)] / (iterations + 1)),
					int(image[addressConverter( y, x, 2)] / (iterations + 1)));

			}
	}
	fclose(ppm);
}
