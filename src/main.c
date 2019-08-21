
#include "common.h"
#include "inference.h"

int main()
{
    char *model_name = "./models/save_model_binary.dat";
    char *filename = "./example.img";
    clock_t before = clock();
    uInference(model_name, filename);
    clock_t difference = clock() - before;
    float msec = difference * 1000.0 / (float)CLOCKS_PER_SEC;
    debug("msec: %f ms", msec);
    return (0);
}