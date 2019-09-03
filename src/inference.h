#ifndef INFERENCE_H
#define INFERENCE_H

#include "common.h"

void conv2d_load_inference(FILE *file, in_out *in);
void bn_load_inference(FILE *file, in_out *in);
void activation_load_inference(FILE *file, in_out *in);
void maxpooling_load_inference(FILE *file, in_out *in);
void flatten_load_inference(FILE *file, in_out *in);
void dense_load_inference(FILE *file, in_out *in);

in_out* uInference(in_out *im, char *model_name);

void yolo_v2(in_out *in, int resize_w, int resize_h);

#endif