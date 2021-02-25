#define WIDTH 1024
#define HEIGHT 512
#define BYTESPERPIXEL 3
#define BLKSIZE 16384
#define THRDSIZE 32
#define OBJNUMBER 10

#define addressConverter(H, W, P) (((W) * (BYTESPERPIXEL)) + ((H) * (WIDTH) * (BYTESPERPIXEL)) + (P))
