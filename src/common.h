#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define debug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)


typedef struct {
    int h;
    int w;
    int c;
    float *data;
} image;

image load_image(char *filename, int w, int h, int c);
void free_image(image m);
void Normalize_image(image* im, float mean, float std);
void print_image(image im);


#endif