//Define images parameters
#define WIDTH 512
#define HEIGHT 256
#define BYTESPERPIXEL 3
#define ANTIALIASINGDEBUG false
#define ITERATIONS 100
#define MAXRECURSIONDEPTH 5
#define OBJNUMBER 10
#define SAVEALLITERATIONS true

//Define the block size for parallelization
#define THRDSIZE 64
#define BLKSIZE WIDTH * HEIGHT / THRDSIZE

//Define textures names
#define TEXONENAME "1.ppm"
#define TEXTWONAME "2.ppm"

//Function to convert from coordinates of the image to memory address
#define addressConverter(H, W, P) (((W) * (BYTESPERPIXEL)) + ((H) * (WIDTH) * (BYTESPERPIXEL)) + (P))

#define PI 3.14159265359
