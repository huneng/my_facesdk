#ifndef _TOOL_H_
#define _TOOL_H_

#include "type_defs.h"

#include <stdint.h>
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#include <android/log.h>
#endif

#include <pthread.h>
#define MAX_THREAD_NUM 1

#define DEAD_TIME 1543630124

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


#define MAX_PTS_SIZE 101
#define MAX_PTS_SIZE2 (MAX_PTS_SIZE << 1)

typedef struct{
    float scale;
    float angle;

    HPoint2f cen1, cen2;
} TranArgs;


typedef struct {
    float pts[MAX_PTS_SIZE2];
    int ptsSize;
} Shape;


void resizer_bilinear_gray(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts);
void extract_area_from_image(uint8_t *img, int width, int height, int stride, uint8_t *patch, HRect &rect);

void affine_image(uint8_t *src, int srcw, int srch, int srcs, HPoint2f cenS,
        uint8_t *dst, int dstw, int dsth, int dsts, HPoint2f cenD, float scale, float angle);


void similarity_transform(Shape &src, Shape &dst, TranArgs &arg);

void affine_shape(Shape &shapeSrc, Shape &shapeDst, TranArgs &arg);
void affine_sample(uint8_t *img, int width, int height, int stride, Shape &shape, TranArgs &arg);

HRect get_shape_rect(Shape &shape);

void bgra2gray(uint8_t *bgraData, int width, int height, int stride, uint8_t *data);
void rgba2gray(uint8_t *bgraData, int width, int height, int stride, uint8_t *data);
void rotate_width_degree(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int &dstw, int &dsth, int &dsts, int degree);

void* aligned_malloc(size_t len, size_t align);
void aligned_free(void* ptr);

#endif
