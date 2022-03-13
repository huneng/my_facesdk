#ifndef _TOOL_H_
#define _TOOL_H_

#include "type_defs.h"

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <assert.h>

typedef struct{
    int x;
    int y;
    int width;
    int height;
}HRect;


typedef struct {
    float x;
    float y;
} HPoint2f;


void resizer_bilinear_gray(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts);
void extract_area_from_image(uint8_t *img, int width, int height, int stride, uint8_t *patch, HRect &rect);

void affine_image(uint8_t *src, int srcw, int srch, int srcs, HPoint2f cenS,
        uint8_t *dst, int dstw, int dsth, int dsts, HPoint2f cenD, float scale, float angle);

void bgra2gray(uint8_t *bgraData, int width, int height, int stride, uint8_t *data);
void rgba2gray(uint8_t *bgraData, int width, int height, int stride, uint8_t *data);
void rotate_width_degree(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int &dstw, int &dsth, int &dsts, int degree);

int read_pts_file(const char *filePath, float *shape, int &ptsSize);
int write_pts_file(const char *filePath, float *shape, int ptsSize);

#endif
