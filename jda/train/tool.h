#ifndef _JDA_TOOL_H_
#define _JDA_TOOL_H_

#include "type_defs.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <assert.h>
#include <omp.h>

#include <sys/time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


typedef struct{
    int x;
    int y;
    int width;
    int height;
}HRect;


int read_file_list(const char *filePath, std::vector<std::string> &fileList);

void analysis_file_path(const char* filePath, char *rootDir, char *fileName, char *ext);

void resizer_bilinear_gray(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts);

void mean_filter_3x3(uint8_t *img, int width, int height, int stride);
void mean_filter_3x3(uint8_t *img, int width, int height, int stride, uint8_t *res);
uint8_t* mean_filter_3x3_res(uint8_t *img, int width, int height, int stride);

void sort_arr_float(float *data, int size);

void transform_image(uint8_t *img, int width, int height, int stride, uint8_t *dImg);
void transform_image(uint8_t *img, int width, int height, int stride);

void write_matrix(double *data, int cols, int rows, const char *outFile);
void write_matrix(float **matrix, int cols, int rows, const char *outFile);
void write_matrix(float *data, int cols, int rows, const char *outFile);

#endif
