#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>

#define debug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)


typedef struct {
    int h;
    int w;
    int c;
    float *data;
} image;

typedef image in_out;

image load_image(char *filename, int w, int h, int c);
void Normalize_image(image* im, float mean, float std);
void print_image(image im);

void free_image(image *m);
void free_in_out(in_out *m);

#endif