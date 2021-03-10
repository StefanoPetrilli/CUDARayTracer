#define WIDTH 2048
#define HEIGHT 1024
#define BYTESPERPIXEL 3
//TODO create the BLKsize and thrdsize automatically based on the dimension fo the image
#define BLKSIZE 32768
#define THRDSIZE 64
#define OBJNUMBER 10
#define ITERATIONS 300

#define addressConverter(H, W, P) (((W) * (BYTESPERPIXEL)) + ((H) * (WIDTH) * (BYTESPERPIXEL)) + (P))
