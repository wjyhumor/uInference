
#include "common.h"
#include "inference.h"

int main()
{
    /*  Object detection */
    char *model_name = "./models_od/mobilenet.dat";
    //char *model_name = "./models_od/tiny_yolo_ocr_7.dat";
    char *filename = "./ex_od.img";
    int resize_w = 320;
    int resize_h = 105;
    in_out im = load_image(filename, resize_w, resize_h, 1);
    normalize_image_255(&im);
    
    /*  Classification 
    char *model_name = "./models_class/save_model.dat";
    char *filename = "./ex_class.img"; 
    int resize_w = 16;
    int resize_h = 16;
    float mean = 122.81543917085412;
    float std = 77.03797602437342;
    in_out im = load_image(filename, resize_w, resize_h, 1);
    normalize_image(&im, mean, std);
    */

    clock_t before = clock();
    in_out* res = uInference(&im, model_name);
    clock_t difference = clock() - before;
    yolo_v2(res, resize_w, resize_h);
    float msec = difference * 1000.0 / (float)CLOCKS_PER_SEC;
    debug("msec: %f ms", msec);

    //save_in_out(*res);

    free_in_out(&im);
    free_in_out(res);
    return (0);
}