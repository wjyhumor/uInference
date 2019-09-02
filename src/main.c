
#include "common.h"
#include "inference.h"

int main()
{

    char *model_name = "./models_od/tiny_yolo_ocr_6.dat";
    char *filename = "./ex_od.img";
/*
    char *model_name = "./models_class/save_model.dat";
    char *filename = "./ex_class.img";
 */
    clock_t before = clock();
    uInference(model_name, filename);
    clock_t difference = clock() - before;
    float msec = difference * 1000.0 / (float)CLOCKS_PER_SEC;
    debug("msec: %f ms", msec);

    return (0);
}