#ifndef _JDA_CASCADE_H_
#define _JDA_CASCADE_H_

#include "forest.h"

typedef struct {
    Shape meanShape;

    int WINW;
    int WINH;

    Forest *forests;
    int ssize;
    int capacity;

    //detect factor
    float sImgScale;
    float eImgScale;
    float sOffScale;
    float eOffScale;

    int layer;
} JDADetector;


int train(JDADetector **rdetector, const char *posFile, const char *negFile, int WINW, int WINH, int stage);

int load(const char *filePath, JDADetector **detector);
int save(const char *filePath, JDADetector *detector);

int predict(JDADetector *detector, uint8_t *img, int stride, Shape &shape, float &score);

void set_detect_factor(JDADetector *detector, float sImgScale, float eImgScale, float sOffScale, float eOffScale, int layer);
int detect(JDADetector *cc, uint8_t *img, int width, int height, int stride, HRect **resRect, Shape **resShapes, float **resScores);

void release(JDADetector **detector);
#endif
