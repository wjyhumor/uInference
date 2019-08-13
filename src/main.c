
#include "common.h"

void read_conv2d_weights(FILE *file)
{
    int w = fgetc(file);
    int h = fgetc(file);
    int in_c = fgetc(file);
    int out_c = fgetc(file);
    int type = fgetc(file); //{"valid":1, "same":2}
    debug("w:%d, h:%d, in_c:%d, out:%d, type:%d", w, h, in_c, out_c, type);
    
    float* weights = calloc(w*h*in_c, sizeof(float));
    float* bias = calloc(1, sizeof(float));
    for(int i = 0; i < w*h*in_c; i++)
    {
        fread(weights+i,sizeof(float),1,file);
    }
    fread(bias,sizeof(float),1,file);
    for(int i = 0; i < w*h*in_c; i++)
    {
        printf("%f ", weights[i]);
    }
    printf("\n");
    debug("%f", *bias);

    


    free(weights);
    free(bias);
}

void uInference(char *model_name, char *filename)
{
    debug("model_name: %s, filename: %s", model_name, filename);
    int resize_w = 16;
    int resize_h = 16; 
    float mean = 122.81543917085412;
    float std = 77.03797602437342;

    // load image and normalize
    image im = load_image(filename, resize_w, resize_h, 1);
    Normalize_image(&im, mean, std);
    //print_image(im);

    // load model and weights and inference
    FILE *file = fopen(model_name, "rb");
    if (file == 0)
    {
        printf("Couldn't open file: %s\n", model_name);
        exit(0);
    }
    
    int layers = fgetc(file);
    debug("layers: %d", layers);
    for(int i = 0; i < layers; i++)
    {
        int layer_type = fgetc(file);
        debug("layer: %d", layer_type);
        switch(layer_type)
        {
            case 1: //Conv2D
                read_conv2d_weights(file);
                break;
            case 2: //BatchNormalization

                break;
            case 3: //Activation

                break;
            case 4: //MaxPooling2D

                break;
            case 5: //Dense

                break;
            default: 
                debug("layer_type: %d not recognized!", layer_type);
                exit(0);
        }
        break;
    }

    // free 
    free_image(im);
}

int main()
{
    char *model_name = "./models/save_model_binary.dat";
    char *filename = "./example.jpg";
    uInference(model_name, filename);
    return(0);
}