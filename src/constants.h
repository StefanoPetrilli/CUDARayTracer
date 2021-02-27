#define WIDTH 1024
#define HEIGHT 512
#define BYTESPERPIXEL 3
#define BLKSIZE 4096
#define THRDSIZE 128
#define OBJNUMBER 10

#define addressConverter(H, W, P) (((W) * (BYTESPERPIXEL)) + ((H) * (WIDTH) * (BYTESPERPIXEL)) + (P))
