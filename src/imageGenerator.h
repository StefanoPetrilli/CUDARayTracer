#include <stdio.h>
#include "constants.h"

/**
 * Function that generates the image
 */
void generate(int width, int height, int bytesPerPixel, float* image){

	FILE *ppm = fopen("./images/output.ppm", "wb");

	//Write the image's header
	fprintf(ppm, "P3\n");
	fprintf(ppm, "%d %d\n255\n", width, height);
	//Append to the file all the pixels
	for (int y = height - 1; y > 0; y--){
			for (int x = 0; x < width; x++) {

				fprintf(ppm, "%d %d %d\n", int(image[addressConverter(y, x, 0)]),
					int(image[addressConverter(y, x, 1)]),
					int(image[addressConverter( y, x, 2)]));

			}
	}
	fclose(ppm);
}
