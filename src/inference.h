#ifndef INFERENCE_H
#define INFERENCE_H

#include "common.h"

#define WEIGHT_ADDR_BASE ((uint32_t)0x0803C000)
static uint32_t address_shift = 0;

void conv2d_load_inference(int type, FILE *file, in_out *in);
void depthwiseconv2d_load_inference(int type, FILE *file, in_out *in);
void bn_load_inference(int type, FILE *file, in_out *in);
void activation_load_inference(int type, FILE *file, in_out *in);
void maxpooling_load_inference(int type, FILE *file, in_out *in);
void flatten_load_inference(int type, FILE *file, in_out *in);
void dense_load_inference(int type, FILE *file, in_out *in);
void leakyrelu_load_inference(int type, FILE *file, in_out *in);
void relu_load_inference(int type, FILE *file, in_out *in);
void zeropadding2d_load_inference(int type, FILE *file, in_out *in);

void yolo_v2(in_out *in, int resize_w, int resize_h);

// type:0-Linux, 1-STM32;
int read_weight(int type, void *buf, size_t size, size_t n, FILE *fp);

in_out* uInference(int type, in_out *im, char *model_name);

#endif