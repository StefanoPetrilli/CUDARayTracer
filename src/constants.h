#define WIDTH 512
#define HEIGHT 256
#define BYTESPERPIXEL 3
#define BLKSIZE 4096
#define THRDSIZE 32
#define OBJNUMBER 10

#define addressConverter(H, W, P) (((W) * (BYTESPERPIXEL)) + ((H) * (WIDTH) * (BYTESPERPIXEL)) + (P))
