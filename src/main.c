
#include "common.h"
#include "inference.h"

void uInference(char *model_name, char *filename)
{
    debug("model_name: %s, filename: %s", model_name, filename);
    int resize_w = 16;
    int resize_h = 16; 
    float mean = 122.81543917085412;
    float std = 77.03797602437342;

    // load image and normalize
    in_out im = load_image(filename, resize_w, resize_h, 1);
    normalize_image(&im, mean, std);
    //print_in_out(im);

    // load model and weights and inference
    FILE *file = fopen(model_name, "rb");
    if (file == 0)
    {
        printf("Couldn't open file: %s\n", model_name);
        exit(0);
    }
    
    int layers = fgetc(file);
    debug("layers: %d", layers);

    in_out *in = &im;
    in_out out = {0, 0, 0, NULL};
    for(int i = 0; i < layers; i++)
    {
        int layer_type = fgetc(file);
        debug("layer type: %d", layer_type);
        switch(layer_type)
        {
            case 1: //Conv2D
                debug("Layer %d: Conv2D", i);
                debug("in: %d x %d x %d", in->w, in->h, in->c);
                conv2d_load_inference(file, in, &out);
                in->c = out.c;
                in->w = out.w; 
                in->h = out.h;
                in->data = out.data;
                out.data = NULL;
                break;
            case 2: //BatchNormalization
                debug("Layer %d: BatchNormalization", i);
                debug("in: %d x %d x %d", in->w, in->h, in->c);
                bn_load_inference(file, in);
                break;
            case 3: //Activation
                debug("Layer %d: Activation", i);
                debug("in: %d x %d x %d", in->w, in->h, in->c);
                activation_load_inference(file, in);
                break;
            case 4: //MaxPooling2D
                debug("Layer %d: MaxPooling2D", i);
                debug("in: %d x %d x %d", in->w, in->h, in->c);
                maxpooling_load_inference(file, in, &out);
                in->c = out.c;
                in->w = out.w; 
                in->h = out.h;
                in->data = out.data;
                out.data = NULL;
                break;
            case 5: //Flatten
                debug("Layer %d: Flatten", i);
                debug("in: %d x %d x %d", in->w, in->h, in->c);
                flatten_load_inference(file, in);
                break;
            case 6: //Dense
                debug("Layer %d: Dense", i);
                debug("in: %d x %d x %d", in->w, in->h, in->c);
                dense_load_inference(file, in, &out);
                in->c = out.c;
                in->w = out.w; 
                in->h = out.h;
                in->data = out.data;
                out.data = NULL;
                break;
            default: 
                debug("layer_type: %d not recognized!", layer_type);
                exit(0);
        }
    }
    fclose(file);
    // free 
    free_in_out(in);
    free_in_out(&out);
    free_in_out(&im);
}

int main()
{
    char *model_name = "./models/save_model_binary.dat";
    char *filename = "./example.img";
    clock_t before = clock();
    uInference(model_name, filename);
    clock_t difference = clock() - before;
    float msec = difference * 1000.0 / (float)CLOCKS_PER_SEC;
    debug("msec: %f ms", msec);
    return(0);
}