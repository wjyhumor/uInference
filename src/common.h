#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define debug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)

typedef struct
{
    int h;
    int w;
    int c;
    float *data;
} in_out;

in_out load_image(char *filename, int w, int h, int c);
void normalize_image(in_out *im, float mean, float std);
void normalize_image_255(in_out *im);
void free_in_out(in_out *m);
void print_in_out(in_out im);
void save_in_out(in_out im);

#endif
