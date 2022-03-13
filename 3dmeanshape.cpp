#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#define PTS_SIZE 68

float MEAN_SHAPE_DATA[] = {
    -73.393524, -72.775017, -70.533638, -66.850060, -59.790188, -48.368973, -34.121101, -17.875410,   0.098749,
     17.477032,  32.648968,  46.372356,  57.343479,  64.388481,  68.212036,  70.486404,  71.375824, -61.119408,
    -51.287586, -37.804798, -24.022755, -11.635713,  12.056636,  25.106256,  38.338589,  51.191006,  60.053852,
      0.653940,   0.804809,   0.992204,   1.226783, -14.772472,  -7.180239,   0.555920,   8.272499,  15.214351,
    -46.047291, -37.674686, -27.883856, -19.648268, -28.272964, -38.082417,  19.265867,  27.894192,  37.437531,
     45.170807,  38.196453,  28.764990, -28.916267, -17.533194,  -6.684590,   0.381001,   8.375443,  18.876617,
     28.794413,  19.057573,   8.956375,   0.381549,  -7.428895, -18.160633, -24.377489,  -6.897633,   0.340663,
      8.444722,  24.474474,   8.449166,   0.205322,  -7.198266, -29.801432, -10.949766,   7.929818,  26.074280,
     42.564388,  56.481079,  67.246994,  75.056892,  77.061287,  74.758446,  66.929024,  56.311390,  42.419125,
     25.455879,   6.990805, -11.666193, -30.365191, -49.361603, -58.769794, -61.996155, -61.033398, -56.686760,
    -57.391033, -61.902187, -62.777714, -59.302345, -50.190254, -42.193790, -30.993721, -19.944595,  -8.414541,
      2.598255,   4.751589,   6.562900,   4.661005,   2.643046, -37.471413, -42.730511, -42.711517, -36.754742,
    -35.134495, -34.919044, -37.032307, -43.342445, -43.110821, -38.086514, -35.532024, -35.484287,  28.612717,
     22.172188,  19.029051,  20.721119,  19.035460,  22.394110,  28.079924,  36.298248,  39.634575,  40.395645,
     39.836407,  36.677898,  28.677771,  25.475977,  26.014269,  25.326199,  28.323008,  30.596216,  31.408737,
     30.844875,  47.667534,  45.909405,  44.842579,  43.141113,  38.635300,  30.750622,  18.456453,   3.609035,
     -0.881698,   5.181201,  19.176563,  30.770571,  37.628628,  40.886311,  42.281448,  44.142567,  47.140427,
     14.254422,   7.268147,   0.442051,  -6.606501, -11.967398, -12.051204,  -7.315098,  -1.022953,   5.349435,
     11.615746, -13.380835, -21.150852, -29.284037, -36.948059, -20.132004, -23.536684, -25.944448, -23.695742,
    -20.858156,   7.037989,   3.021217,   1.353629,  -0.111088,  -0.147273,   1.476612,  -0.665746,   0.247660,
      1.696435,   4.894163,   0.282961,  -1.172675,  -2.240310, -15.934335, -22.611355, -23.748438, -22.721994,
    -15.610679,  -3.217393, -14.987997, -22.554245, -23.591625, -22.406107, -15.121907,  -4.785684, -20.893742,
    -22.220478, -21.025520,  -5.712776, -20.671490, -21.903669, -20.328022,
};


#define HU_SWAP(x, y, type) {type tmp = (x); (x) = (y); (y) = (tmp);}
#define HU_MIN(i, j) ((i) > (j) ? (j) : (i))
#define HU_MAX(i, j) ((i) < (j) ? (j) : (i))
#define HU_PI 3.1415926535


char show_shape(float *shape, int ptsSize, const char *winName){
    cv::Mat img;

    float minx = shape[0];
    float maxx = shape[0];
    float miny = shape[ptsSize];
    float maxy = shape[ptsSize];
    float cx, cy;

    for(int i = 1; i < ptsSize; i++){
        minx = HU_MIN(minx, shape[i]);
        maxx = HU_MAX(maxx, shape[i]);

        miny = HU_MIN(miny, shape[i + ptsSize]);
        maxy = HU_MAX(maxy, shape[i + ptsSize]);
    }

    cx = (maxx + minx) / 2;
    cy = (maxy + miny) / 2;

    int width = ceil(maxx - minx + 9);
    int height = ceil(maxy - miny + 9);

    int deltax = width / 2 - cx;
    int deltay = height / 2 - cy;

    img = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    for(int i = 0; i < ptsSize; i++)
        cv::circle(img, cv::Point2f(shape[i] + deltax, shape[i + ptsSize] + deltay), 2, cv::Scalar(0, 255, 0), -1);

    cv::imshow(winName, img);
    return cv::waitKey();
}


void shape_3d_to_2d(float *shape3D, float *shape2D, int ptsSize){
    int n = 0;
    for(int i = 0; i < ptsSize; i++){
        shape2D[i] = shape3D[i];
        shape2D[i + ptsSize] = shape3D[i + ptsSize];
    }
}


void transform_shape(cv::Mat_<float> &shape, float ex, float ey, float ez, cv::Mat_<float>& res){
    assert(shape.rows == 3 && shape.cols == PTS_SIZE);

    cv::Mat_<float> shapeMat(shape.rows + 1, shape.cols);
    int ptsSize = shape.cols;

    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    cv::Mat_<float> eyeMat = cv::Mat_<float>::eye(4, 4);
    cv::Mat_<float> exMat, eyMat, ezMat, rotateMat;
    cv::Mat_<float> resMat;
    float sina, cosa;

    shapeMat.setTo(0.0);

    shapeMat.row(0) += shape.row(0);
    shapeMat.row(1) += shape.row(1);
    shapeMat.row(2) += shape.row(2);
    shapeMat.row(3) += 1;

    cx = 0, cy = 0, cz = 0;
    for(int i = 0; i < ptsSize; i++){
        cx += shapeMat(0, i);
        cy += shapeMat(1, i);
        cz += shapeMat(2, i);
    }

    cx /= ptsSize;
    cy /= ptsSize;
    cz /= ptsSize;

    shapeMat.row(0) -= cx;
    shapeMat.row(1) -= cy;
    shapeMat.row(2) -= cz;

    eyeMat.copyTo(exMat);
    eyeMat.copyTo(eyMat);
    eyeMat.copyTo(ezMat);

    sina = sin(ex), cosa = cos(ex);
    exMat(1, 1) = cosa, exMat(1, 2) = -sina;
    exMat(2, 1) = sina, exMat(2, 2) =  cosa;

    sina = sin(ey), cosa = cos(ey);
    eyMat(0, 0) =  cosa, eyMat(0, 2) = sina;
    eyMat(2, 0) = -sina, eyMat(2, 2) = cosa;

    sina = sin(ez), cosa = cos(ez);
    ezMat(0, 0) = cosa, ezMat(0, 1) = -sina;
    ezMat(1, 0) = sina, ezMat(1, 1) =  cosa;

    rotateMat = exMat * eyMat * ezMat;

    resMat = rotateMat * shapeMat;

    res.create(3, ptsSize);
    res.setTo(0.0);

    res.row(0) += resMat.row(0) + cx;
    res.row(1) += resMat.row(1) + cy;
    res.row(2) += resMat.row(2) + cz;
}


cv::Mat draw_shape(float *shape, int ptsSize){
    cv::Mat img;

    float minx = shape[0];
    float maxx = shape[0];
    float miny = shape[ptsSize];
    float maxy = shape[ptsSize];
    float cx, cy;

    for(int i = 1; i < ptsSize; i++){
        minx = HU_MIN(minx, shape[i]);
        maxx = HU_MAX(maxx, shape[i]);

        miny = HU_MIN(miny, shape[i + ptsSize]);
        maxy = HU_MAX(maxy, shape[i + ptsSize]);
    }

    cx = (maxx + minx) / 2;
    cy = (maxy + miny) / 2;

    int width = ceil(maxx - minx + 9);
    int height = ceil(maxy - miny + 9);

    int deltax = width / 2 - cx;
    int deltay = height / 2 - cy;

    img = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    for(int i = 0; i < ptsSize; i++)
        cv::circle(img, cv::Point2f(shape[i] + deltax, shape[i + ptsSize] + deltay), 2, cv::Scalar(0, 255, 0), -1);

    return img;
}


void get_offset(float *shape, int ptsSize, float &dx, float &dy, int winSize){

    float cx1 = 0.0f, cy1 = 0.0f;
    float cx2 = 0.0f, cy2 = 0.0f;

    for(int i = 0; i < 17; i++){
        cx1 += shape[i];
        cy1 += shape[i + ptsSize];
    }

    cx1 /= 17;
    cy1 /= 17;

    for(int i = 17; i < ptsSize; i++){
        cx2 += shape[i];
        cy2 += shape[i + ptsSize];
    }

    cx2 /= (ptsSize - 17);
    cy2 /= (ptsSize - 17);

    dx = (cx2 - cx1) / winSize;
    dy = (cy2 - cy1) / winSize + 0.3;
}


int main(int argc, char **argv){

    float res[136];
    cv::Mat_<float> meanShape3D(3, PTS_SIZE, MEAN_SHAPE_DATA);

    shape_3d_to_2d((float*)meanShape3D.data, res, PTS_SIZE);
    //show_shape(res, PTS_SIZE, "meanshape");

    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;

    for(int i = 0; i < PTS_SIZE; i++){
        minx = HU_MIN(minx, res[i]);
        maxx = HU_MAX(maxx, res[i]);

        miny = HU_MIN(miny, res[i + PTS_SIZE]);
        maxy = HU_MAX(maxy, res[i + PTS_SIZE]);
    }

    int winSize = HU_MAX(maxx - minx + 1, maxy - miny + 1);

    cv::Mat_<float> transRes;

    int len = 36;
    for(int i = 0; i <= len; i++){
        float ex = 0;
        float ey = HU_PI / len * i - HU_PI / 2;
        float ez = 0;

        //printf("angle = %f, %f, %f\n", ex * 180.0 / HU_PI, ey * 180.0 / HU_PI, ez * 180.0 / HU_PI);
        transform_shape(meanShape3D, ex, ey, ez, transRes);

        shape_3d_to_2d((float*)transRes.data, res, PTS_SIZE);

        float dx, dy;

        get_offset(res, PTS_SIZE, dx, dy, winSize);

        printf("%7.3f %9f %9f\n", ey * 180.0f / HU_PI, dx, dy);
    }

    return 0;
}
