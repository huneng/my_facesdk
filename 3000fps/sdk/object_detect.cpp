#include "object_detect.h"

#define QT_VERSION_EPCH 1
#define QT_VERSION_MAJOR 0
#define QT_VERSION_MINOR 0

#define QT_SWAP(x, y, type) {type tmp = (x); (x) = (y); (y) = (tmp);}
#define QT_MIN(i, j) ((i) > (j) ? (j) : (i))
#define QT_MAX(i, j) ((i) < (j) ? (j) : (i))
#define QT_ABS(a) ((a) < 0 ? (-a) : (a))


#define EPSILON 0.000001f
#define QT_PI 3.1415926535

#ifndef QT_FREE
#define QT_FREE(arr) \
{ \
    if(arr != NULL) \
        free(arr); \
    arr = NULL; \
}
#endif



static void QT_integral_image(uint8_t *img, int width, int height, int stride, uint32_t *iImg, int istride){

    uint32_t *ptrLine1 = iImg;
    uint32_t *ptrLine2 = iImg + istride;

    for(int x = 0; x < width; x++)
        iImg[x] = iImg[x - 1] + img[x];

    img += stride;

    for(int y = 1; y < height; y ++){
        uint32_t sum = 0;

        for(int x = 0; x < width; x++){
            sum += img[x];
            ptrLine2[x] = ptrLine1[x] + sum;
        }

        img += stride;
        ptrLine1 += istride;
        ptrLine2 += istride;
    }
}


#define QT_FIX_POINT_Q 14

static void QT_resize_gray_image(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts){
    uint16_t *xtable = NULL;

    uint16_t FIX_0_5 = 1 << (QT_FIX_POINT_Q - 1);
    float scalex, scaley;

    scalex = srcw / float(dstw);
    scaley = srch / float(dsth);

    xtable = new uint16_t[dstw * 2];

    for(int x = 0; x < dstw; x++){
        float xc = x * scalex;

        if(xc < 0) xc = 0;
        if(xc >= srcw - 1) xc = srcw - 1.01f;

        int x0 = int(xc);

        xtable[x * 2] = x0;
        xtable[x * 2 + 1] = (1 << QT_FIX_POINT_Q) - (xc - x0) * (1 << QT_FIX_POINT_Q);
    }

    int sId = 0, dId = 0;

    for(int y = 0; y < dsth; y++){
        int x;
        float yc;

        uint16_t wy0;
        uint16_t y0, y1;
        uint16_t *ptrTab = xtable;

        yc = y * scaley;

        if(yc < 0) yc = 0;
        if(yc >= srch - 1) yc = srch - 1.01f;

        y0 = uint16_t(yc);
        y1 = y0 + 1;

        wy0 = (1 << QT_FIX_POINT_Q) - uint16_t((yc - y0) * (1 << QT_FIX_POINT_Q));

        sId = y0 * srcs;

        uint8_t *ptrDst = dst + dId;

        for(x = 0; x <= dstw - 4; x += 4){
            uint16_t x0, x1, wx0;
			int vy0, vy1;
            uint8_t *ptrSrc0, *ptrSrc1;

            //1
            x0 = ptrTab[0], x1 = x0 + 1;
            wx0 = ptrTab[1];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
            vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			ptrDst[0] = (wy0 * (vy0 - vy1) + (vy1 << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            //2
            x0 = ptrTab[2], x1 = x0 + 1;
            wx0 = ptrTab[3];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

			vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			ptrDst[1] = (wy0 * (vy0 - vy1) + (vy1 << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            //3
            x0 = ptrTab[4], x1 = x0 + 1;
            wx0 = ptrTab[5];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

			vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			ptrDst[2] = (wy0 * (vy0 - vy1) + (vy1 << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            //4
            x0 = ptrTab[6], x1 = x0 + 1;
            wx0 = ptrTab[7];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

			vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
			ptrDst[3] = (wy0 * (vy0 - vy1) + (vy1 << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            ptrDst += 4;
            ptrTab += 8;
        }

        for(; x < dstw; x++){
            uint16_t x0, x1, wx0, vy0, vy1;

            uint8_t *ptrSrc0, *ptrSrc1;
            x0 = ptrTab[0], x1 = x0 + 1;
            wx0 = ptrTab[1];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;
            vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            dst[y * dsts + x] = (wy0 * (vy0 - vy1) + (vy1 << QT_FIX_POINT_Q) + FIX_0_5) >> QT_FIX_POINT_Q;

            ptrTab += 2;
        }

        dId += dsts;
    }


    delete [] xtable;
}


#define GET_VALUE(ft, img, stride, value) \
{ \
    int x0, y0, x1, y1;                   \
    int a, b;                             \
    x0 = ft->x00;                         \
    y0 = ft->y00 * stride;                \
    x1 = ft->x01;                         \
    y1 = ft->y01 * stride;                \
                                          \
    a = img[y1 + x1] - img[y1 + x0] - img[y0 + x1] + img[y0 + x0]; \
                                          \
    x0 = ft->x10;                         \
    y0 = ft->y10 * stride;                \
    x1 = ft->x11;                         \
    y1 = ft->y11 * stride;                \
                                          \
    b = img[y1 + x1] - img[y1 + x0] - img[y0 + x1] + img[y0 + x0]; \
    value = (a - b); \
}



float predict(Tree *root, int depth, uint32_t *iImg, int istride){
    for(int i = 0; i < depth - 1; i++){
        float feat;
        FeatTemp *ptrFeat = &root->ft;
        GET_VALUE(ptrFeat, iImg, istride, feat);
        root = root->lchild + (feat > root->thresh);
    }

    assert(root != NULL);

    return root->score;
}



void load(Tree* root, int depth, FILE *fin){

    assert(root != NULL && fin != NULL);

    int nlSize = (1 << depth) - 1;
    int leafSize = 1 << (depth - 1);
    int nodeSize = nlSize - leafSize;
    int ret;

    for(int i = 0; i < nlSize; i++){
        Node *node = root + i;

        if(i < nodeSize){
            ret = fread(&node->thresh, sizeof(float), 1, fin); assert(ret == 1);
            ret = fread(&node->ft, sizeof(FeatTemp), 1, fin); assert(ret == 1);

            node->lchild = root + i * 2 + 1;
            node->rchild = root + i * 2 + 2;
        }
        else{
            ret = fread(&node->score, sizeof(float), 1, fin); assert(ret == 1);
            node->lchild = NULL;
            node->rchild = NULL;
        }
    }
}


int predict(StrongClassifier *sc, uint32_t *intImg, int istride, float &score){
    int i = 0;

    score = 0;

    for(i = 0; i < sc->treeSize; i++){
        score += predict(sc->trees[i], sc->depth, intImg, istride);
        if(score <= sc->threshes[i])
            return 0;
    }

    return 1;
}


int predict(StrongClassifier *sc, int scSize, uint32_t *intImg, int istride, float &score){
    score = 0;
    for(int i = 0; i < scSize; i++){
        StrongClassifier *ptrSC = sc + i;

        Tree **ptrTrees = ptrSC->trees;
        int depth = ptrSC->depth;
        int treeSize = ptrSC->treeSize;

        float sscore = 0;
        int j;

        for(j = 0; j < treeSize; j++){
            sscore += predict(ptrTrees[j], depth, intImg, istride);
            if(sscore <= ptrSC->threshes[j]){
                return 0;
            };
        }

        score = sscore;
    }

    return 1;
}


int load(StrongClassifier *sc, FILE *fin){
    if(fin == NULL || sc == NULL){
        return 1;
    }

    int ret;
    int nlSize;

    ret = fread(&sc->treeSize, sizeof(int), 1, fin); assert(ret == 1);

    sc->trees = new Tree*[sc->treeSize];
    sc->threshes = new float[sc->treeSize];

    ret = fread(&sc->depth, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(sc->threshes, sizeof(float), sc->treeSize, fin); assert(ret == sc->treeSize);


    nlSize = (1 << sc->depth) - 1;

    sc->capacity  = sc->treeSize;

    for(int i = 0; i < sc->treeSize; i++){
        sc->trees[i] = new Node[nlSize];
        memset(sc->trees[i], 0, sizeof(Node) * nlSize);
        load(sc->trees[i], sc->depth, fin);
    }

    return 0;
}


void release_data(StrongClassifier *sc){
    if(sc != NULL){
        if(sc->trees != NULL){
            for(int i = 0; i < sc->treeSize; i++){
                if(sc->trees[i] != NULL)
                    delete [] sc->trees[i];
                sc->trees[i] = NULL;
            }

            delete [] sc->trees;
        }

        sc->trees = NULL;

        if(sc->threshes != NULL)
            delete [] sc->threshes;

        sc->threshes = NULL;
    }

    sc->capacity = 0;
    sc->treeSize = 0;
}


void release(StrongClassifier **sc){
    if(sc == NULL)
        return;

    release_data(*sc);

    delete *sc;
    *sc = NULL;
}


int predict(QTObjectDetector *cc, uint32_t *iImg, int iStride, float &score){

    score = 0;

    for(int i = 0; i < cc->ssize; i++){
        float t;
        if(predict(cc->sc + i, iImg, iStride, t) == 0)
            return 0;

        score = t;
    }

    return 1;
}


int predict(QTObjectDetector *cc, uint8_t *img, int width, int height, int stride)
{
    assert(width == cc->WINW && height == cc->WINH);

    uint32_t iImgBuf[65 * 65];
    uint32_t *iImg = iImgBuf + cc->WINW + 1 + 1;
    float score;
    int treeSize, ret;

    memset(iImgBuf, 0, sizeof(uint32_t) * (cc->WINW + 1) * (cc->WINH + 1));

    QT_integral_image(img, width, height, stride, iImg, cc->WINW + 1);

    treeSize = cc->sc->treeSize;
    cc->sc->treeSize = 32;
    ret = predict(cc->sc, iImg, cc->WINW + 1, score);
    cc->sc->treeSize = treeSize;

    return ret;
}


void init_detect_factor(QTObjectDetector *cc, float startScale, float endScale, float offset, int layer){
    cc->startScale = startScale;
    cc->endScale = endScale;
    cc->layer = layer;
    cc->offsetFactor = offset;

    float stepFactor = powf(endScale / startScale, 1.0f / layer);
}


#define MERGE_RECT



int calc_overlapping_area(HRect &rect1, HRect &rect2){
    int cx1 = rect1.x + rect1.width / 2;
    int cy1 = rect1.y + rect1.height / 2;
    int cx2 = rect2.x + rect2.width / 2;
    int cy2 = rect2.y + rect2.height / 2;

    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;

    if(abs(cx1 - cx2) < rect1.width / 2 + rect2.width/2 && abs(cy1 - cy2) < rect1.height / 2 + rect2.height / 2){
        x0 = QT_MAX(rect1.x , rect2.x);
        x1 = QT_MIN(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
        y0 = QT_MAX(rect1.y, rect2.y);
        y1 = QT_MIN(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);
    }
    else {
        return 0;
    }

    return (y1 - y0 + 1) * (x1 - x0 + 1);
}


int merge_rects(HRect *rects, float *confs, int size){
    if(size < 2) return size;

    uint8_t *flags = new uint8_t[size];

    memset(flags, 0, sizeof(uint8_t) * size);

    for(int i = 0; i < size; i++){
        if(flags[i] == 1)
            continue;

        float area0 = 1.0f / (rects[i].width * rects[i].height);

        for(int j = i + 1; j < size; j++){
            if(flags[j] == 1) continue;

            float area1 = 1.0f / (rects[j].width * rects[j].height);

            int overlap = calc_overlapping_area(rects[i], rects[j]);

            if(overlap * area1 > 0.6f || overlap * area0 > 0.6f){
                if(confs[i] > confs[j])
                    flags[j] = 1;
                else
                    flags[i] = 1;
            }
        }
    }

    for(int i = 0; i < size; i++){
        if(flags[i] == 0) {
            continue;
        }

        flags[i] = flags[size - 1];

        rects[i] = rects[size - 1];
        confs[i] = confs[size - 1];

        i --;
        size --;
    }

    delete []flags;
    flags = NULL;

    return size;
}


#define FIX_Q 14

int detect_one_scale(QTObjectDetector *cc, float scale, uint32_t *iImg, int width, int height, int stride, HRect *resRect, float *resScores){
    int WINW = cc->WINW;
    int WINH = cc->WINH;

    int dx = WINW * cc->offsetFactor;
    int dy = WINH * cc->offsetFactor;

    int count = 0;
    float score;

    int HALF_ONE = 1 << (FIX_Q - 1);
    int FIX_SCALE = scale * (1 << FIX_Q);

    for(int y = 0; y <= height - WINH; y += dy){
        for(int x = 0; x <= width - WINW; x += dx){
            uint32_t *ptr = iImg + y * stride + x;

            if(predict(cc->sc, cc->ssize, iImg + y * stride + x, stride, score) == 1){
                resRect[count].x = (x * FIX_SCALE + HALF_ONE) >> FIX_Q;
                resRect[count].y = (y * FIX_SCALE + HALF_ONE) >> FIX_Q;

                resRect[count].width = (WINW * FIX_SCALE + HALF_ONE) >> FIX_Q;
                resRect[count].height = (WINH * FIX_SCALE + HALF_ONE) >> FIX_Q;

                resScores[count] = score;
                count++;
            }
        }
    }

#ifdef MERGE_RECT
    count = merge_rects(resRect, resScores, count);
#endif

    return count;
}


int calculate_max_size(int width, int height, float startScale, int winSize){
    int minwh = QT_MIN(width, height);

    assert(startScale < 1.0f);

    int size = minwh * startScale;
    float scale = (float)winSize / size;

    width ++;
    height ++;

    if(scale < 1)
        return width * height;

    return (width * scale + 0.5f) * (height * scale + 0.5f);
}


int detect(QTObjectDetector *cc, uint8_t *img, int width, int height, int stride, HRect **resRect, float **rscores){
    int WINW, WINH, capacity;
    float scale, stepFactor;

    uint8_t *dImg, *ptrSrc, *ptrDst;
    uint32_t *iImgBuf, *iImg;

    HRect *rects;
    float *scores;
    int top = 0;

    int srcw, srch, srcs, dstw, dsth, dsts;
    int minSide;
    int count;

    WINW = cc->WINW;
    WINH = cc->WINH;

    scale = cc->startScale;
    stepFactor = powf(cc->endScale / cc->startScale, 1.0f / (cc->layer));

    capacity = calculate_max_size(width, height, scale, QT_MAX(WINW, WINH));

    dImg = new uint8_t[capacity * 2]; assert(dImg != NULL);
    iImgBuf = new uint32_t[capacity * 2]; assert(iImgBuf != NULL);

    const int BUFFER_SIZE = 1000;
    rects  = new HRect[BUFFER_SIZE]; assert(rects != NULL);
    scores = new float[BUFFER_SIZE]; assert(scores != NULL);

    memset(rects, 0, sizeof(HRect)  * BUFFER_SIZE);
    memset(scores, 0, sizeof(float) * BUFFER_SIZE);

    ptrSrc = img;
    ptrDst = dImg;

    srcw = width;
    srch = height;
    srcs = stride;

    count = 0;

    minSide = QT_MIN(width, height);

    for(int i = 0; i < cc->layer; i++){
        float scale2 = QT_MIN(WINW, WINH) / (minSide * scale);

        dstw = scale2 * width;
        dsth = scale2 * height;
        dsts = dstw;

        QT_resize_gray_image(ptrSrc, srcw, srch, srcs, ptrDst, dstw, dsth, dsts);

        assert(dstw * dsth < 16777216);

        memset(iImgBuf, 0, sizeof(uint32_t) * (dstw + 1) * (dsth + 1));
        iImg = iImgBuf + dstw + 1 + 1;

        QT_integral_image(ptrDst, dstw, dsth, dsts, iImg, dstw + 1);

        count += detect_one_scale(cc, 1.0f / scale2, iImg, dstw, dsth, dstw + 1, rects + count, scores + count);

        ptrSrc = ptrDst;

        srcw = dstw;
        srch = dsth;
        srcs = dsts;

        if(ptrDst == dImg)
            ptrDst = dImg + dstw * dsth;
        else
            ptrDst = dImg;

        scale *= stepFactor;
    }

    if(count > 0){
#ifdef MERGE_RECT
        count = merge_rects(rects, scores, count);
#endif

        *resRect = new HRect[count]; assert(resRect != NULL);
        memcpy(*resRect, rects, sizeof(HRect) * count);
        *rscores = new float[count]; assert(rscores != NULL);
        memcpy(*rscores, scores, sizeof(float) * count);
    }

    delete [] dImg;
    delete [] iImgBuf;
    delete [] rects;
    delete [] scores;

    return count;
}


int load(QTObjectDetector **rcc, const char *filePath){

    FILE *fin = fopen(filePath, "rb");
    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 2;
    }

    QTObjectDetector *cc = new QTObjectDetector;
    memset(cc, 0, sizeof(QTObjectDetector));

    int ret;

    char str[100];
    int versionEpoch, versionMajor, versionMinor;

    ret = fread(str, sizeof(char), 100, fin); assert(ret == 100);
    sscanf(str, "HANGZHOU QIANTU TECHNOLOGY FACE DETECTOR: %d.%d.%d", &versionEpoch, &versionMajor, &versionMinor);

    assert(versionEpoch == QT_VERSION_EPCH);
    assert(versionMajor == QT_VERSION_MAJOR);
    assert(versionMinor == QT_VERSION_MINOR);

    ret = fread(&cc->ssize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&cc->WINW, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&cc->WINH, sizeof(int), 1, fin); assert(ret == 1);

    cc->sc = new StrongClassifier[cc->ssize]; assert(cc->sc != NULL);

    memset(cc->sc, 0, sizeof(StrongClassifier) * cc->ssize);

    for(int i = 0; i < cc->ssize; i++){
        ret = load(cc->sc + i, fin);
        if(ret != 0){
            printf("Load strong classifier error\n");
            fclose(fin);

            delete [] cc->sc;
            delete cc;

            return 2;
        }
    }

    fclose(fin);

    *rcc = cc;

    return 0;
}


void release_data(QTObjectDetector *cc){
    if(cc->sc == NULL)
        return;

    for(int i = 0; i < cc->ssize; i++)
        release_data(cc->sc + i);

    delete [] cc->sc;
    cc->sc = NULL;
}


void release(QTObjectDetector **cc){
    if(*cc == NULL)
        return;

    release_data(*cc);
    delete *cc;
    *cc = NULL;
}
