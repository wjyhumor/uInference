#ifndef INFERENCE_H
#define INFERENCE_H

#include "common.h"

void conv2d_load_inference(FILE *file, in_out *in, in_out *out);
void bn_load_inference(FILE *file, in_out *in);
void activation_load_inference(FILE *file, in_out *in);
void maxpooling_load_inference(FILE *file, in_out *in, in_out *out);
void flatten_load_inference(FILE *file, in_out *in);
void dense_load_inference(FILE *file, in_out *in, in_out *out);

void uInference(char *model_name, char *filename);

#endif