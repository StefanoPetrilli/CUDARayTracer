#define WIDTH 1024
#define HEIGHT 512
#define BYTESPERPIXEL 3
#define BLKSIZE 256
#define THRDSIZE 100

#define addressConverter(H, W, P) (((W) * (BYTESPERPIXEL)) + ((H) * (WIDTH) * (BYTESPERPIXEL)) + (P))
