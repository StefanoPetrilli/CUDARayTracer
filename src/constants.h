#define WIDTH 4096
#define HEIGHT 2048
#define BYTESPERPIXEL 3
#define BLKSIZE 256
#define THRDSIZE 100

#define addressConverter(H, W, P) (((W) * (BYTESPERPIXEL)) + ((H) * (WIDTH) * (BYTESPERPIXEL)) + (P))
