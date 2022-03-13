#include "aligner.h"

#define RADIUS 0.8
#define TREE_DEPTH 4
#define TREE_SIZE 101
#define MAX_TREE_SIZE 1024

#define MIRROR 0
#define TIMES 4


static void get_shape_center(float *shape, int ptsSize, HPoint2f &center){
    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;

    for(int i = 0; i < ptsSize; i++){
        float x = shape[i];
        float y = shape[i + ptsSize];

        minx = HU_MIN(minx, x);
        maxx = HU_MAX(maxx, x);

        miny = HU_MIN(miny, y);
        maxy = HU_MAX(maxy, y);
    }

    center.x = 0.5f * (minx + maxx);
    center.y = 0.5f * (miny + maxy);
}


static float feature_distance(float *feats1, float *feats2, int size){
    float value = 0;

    for(int i = 0; i < size; i++)
        value += (feats1[i] != feats2[i]);

    printf("%f\n", value / size);
    return value / size;
}


void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, HRect &srect, Shape &curShape){
    TranArgs arg;
    HRect rect;

    float scale;

    int ptsSize = aligner->ptsSize;
    int ptsSize2 = ptsSize << 1;
    int winSize = SAMPLE_IMAGE_SIZE * 2 + 1;

    uint8_t *feats = new uint8_t[MAX_TREE_SIZE];

    uint8_t *patch, *src;

    assert(aligner != NULL && img != NULL);

    rect.x = srect.x - (srect.width  >> 1);
    rect.y = srect.y - (srect.height >> 1);
    rect.width  = srect.width << 1;
    rect.height = srect.height << 1;

    patch = new uint8_t[rect.width * rect.height + winSize * winSize];
    src = patch + rect.width * rect.height;

    extract_area_from_image(img, width, height, stride, patch, rect);

    scale = (float)rect.width / winSize;

    resizer_bilinear_gray(patch, rect.width, rect.height, rect.width, src, winSize, winSize, winSize);

    curShape = aligner->meanShape;

    for(int p = 0; p < ptsSize2; p++)
        curShape.pts[p] += SAMPLE_IMAGE_SIZE;

    for(int s = 0; s < aligner->stage; s++){
        Forest *forest = aligner->forests[s];

        memcpy(patch, src, sizeof(uint8_t) * winSize * winSize);

        similarity_transform(curShape, aligner->meanShape, arg);
        arg.cen2.x = winSize >> 1;
        arg.cen2.y = winSize >> 1;

        affine_sample(patch, winSize, winSize, winSize, curShape, arg);

        predict(forest, patch, winSize, winSize, winSize, curShape);

        arg.scale = 1.0f / arg.scale;
        arg.angle = -arg.angle;

        HU_SWAP(arg.cen1, arg.cen2, HPoint2f);
        affine_shape(curShape, curShape, arg);
    }

    for(int i = 0, j = ptsSize; i < ptsSize; i++, j++){
        curShape.pts[i] = curShape.pts[i] * scale + rect.x;
        curShape.pts[j] = curShape.pts[j] * scale + rect.y;
    }

    delete [] patch;
    delete [] feats;
}


void predict(Aligner *aligner, uint8_t *img, int width, int height, int stride, Shape &curShape){
    HRect rect;

    rect = get_shape_rect(curShape);

    predict(aligner, img, width, height, stride, rect, curShape);
}


int load(const char *filePath, Aligner **raligner){
    if(filePath == NULL)
        return 1;

    FILE *fp = fopen(filePath, "rb");

    if(fp == NULL){
        printf("Can't save file %s\n", filePath);
        return 2;
    }

    int ret;
    Aligner *aligner = new Aligner;

    ret = fread(&aligner->stage, sizeof(int), 1, fp); assert(ret == 1);
    ret = fread(&aligner->ptsSize, sizeof(int), 1, fp); assert(ret == 1);

    ret = fread(&aligner->meanShape, sizeof(Shape), 1, fp); assert(ret == 1);

    aligner->forests = new Forest*[aligner->stage];

    for(int i = 0; i < aligner->stage; i++){
        load(fp, &aligner->forests[i]);
    }

    fclose(fp);

    *raligner = aligner;

    return 0;
}


void release_data(Aligner *aligner){
    if(aligner->stage == 0 || aligner->forests == NULL)
        return;

    for(int i = 0; i < aligner->stage; i++){
        release(&aligner->forests[i]);
    }

    delete [] aligner->forests;

    aligner->stage = 0;
    aligner->forests = NULL;
}


void release(Aligner **aligner){
    if(*aligner == NULL)
        return;

    release_data(*aligner);
    delete *aligner;
    *aligner = NULL;
}
