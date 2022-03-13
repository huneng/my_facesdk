#ifndef _FACE_MANAGER_H_
#define _FACE_MANAGER_H_


#include "aligner.h"
#include "object_detect.h"

#define MAX_FACE_SIZE 3


#define FT_IMAGE_NV21 0
#define FT_IMAGE_BGRA 1
#define FT_IMAGE_GRAY 2
#define FT_IMAGE_RGBA 3
#define FT_IMAGE_YUV420 4

#define ROATE_FLAG_0   0
#define ROATE_FLAG_90  1
#define ROATE_FLAG_180 2
#define ROATE_FLAG_270 3

#define PTS_SIZE 101
#define PTS_SIZE2 (PTS_SIZE << 1)

typedef struct {
    Aligner *aligner;
    Aligner *tracker;

    ObjectDetector *detector;

    int tflag;

    Shape lastShapes[MAX_FACE_SIZE];

    int lastSize;
    int ptsSize;
} FaceManager;


int load(const char *detectModel, const char *alignModel, const char *trackModel, FaceManager **rmanager);
int detect_face(FaceManager *manager, uint8_t *img, int width, int height, int stride, int flag, HRect **resRect);
int align_face(FaceManager *manager, uint8_t *img, int width, int height, int stride, Shape **shapes);
int track_one_face(FaceManager *manager, uint8_t *img, int width, int height, int stride, Shape &resShape);
int track_mul_face(FaceManager *manager, uint8_t *img, int width, int height, int stride, Shape *resShapes);

void release(FaceManager **manager);

#ifdef WIN32

#ifdef _MSC_VER

#ifdef __cplusplus
#define WIN_EXPORT extern "C" _declspec(dllexport)

#else //start __cplusplus
#define WIN_EXPORT __declspec(dllexport)

#endif

#endif

#else //start WIN32
#define WIN_EXPORT

#endif

WIN_EXPORT int load_models(const char *detectModel, const char *alignModel, const char *trackModel);

WIN_EXPORT int process_align(uint8_t *img, int width, int height, int stride, int format, float **resShape);
WIN_EXPORT int process_track(uint8_t *img, int width, int height, int stride, int format, int rflag, float resPts[]);

WIN_EXPORT void get_angles(float oriShape[], int ptsSize, float &ex, float &ey, float &ez);

WIN_EXPORT void release_models();

#endif
