#include "face_manager.h"


int load(const char *detectModel, const char *alignModel, const char *trackModel, FaceManager **rmanager)
{
    FaceManager *manager;

    if(detectModel == NULL){
        printf("Can't find detect model\n");
        return 1;
    }

    if(alignModel == NULL && trackModel == NULL){
        printf("Can't find align model or track model\n");
        return 2;
    }

    manager = new FaceManager;

    memset(manager, 0, sizeof(FaceManager));

    if(load(&manager->detector, detectModel) != 0){
        printf("Can't open detector %s\n", detectModel);
        return 3;
    }

    if(alignModel != NULL){
        int ret = load(alignModel, &manager->aligner);
        if(ret != 0){
            printf("Can't open align model %s\n", alignModel);
            return 4;
        }
        manager->ptsSize = manager->aligner->ptsSize;
    }

    if(trackModel != NULL){
        int ret = load(trackModel, &manager->tracker);
        if(ret != 0){
            printf("Can't open track model %s\n", trackModel);
            return 5;
        }
        manager->ptsSize = manager->tracker->ptsSize;
    }

    manager->tflag = 1;

    *rmanager = manager;

    return 0;
}


static int calc_overlapping_area(HRect &rect1, HRect &rect2){
    int cx1 = rect1.x + (rect1.width >> 1);
    int cy1 = rect1.y + (rect1.height >> 1);
    int cx2 = rect2.x + (rect2.width >> 1);
    int cy2 = rect2.y + (rect2.height >> 1);

    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;

    if(abs(cx1 - cx2) < ((rect1.width + rect2.width) >> 1) && abs(cy1 - cy2) < ((rect1.height + rect2.height ) >> 1)){
        x0 = HU_MAX(rect1.x , rect2.x);
        x1 = HU_MIN(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
        y0 = HU_MAX(rect1.y, rect2.y);
        y1 = HU_MIN(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);
    }
    else {
        return 0;
    }

    return (y1 - y0 + 1) * (x1 - x0 + 1);
}


int detect_face(FaceManager *manager, uint8_t *img, int width, int height, int stride, int flag, HRect **resRect){
    QTObjectDetector *detector = manager->detector;
    HRect *rects;
    float *confs;
    int rsize;

    if(detector == NULL)
        return 0;

    rsize = detect(detector, img, width, height, stride, &rects, &confs);

    if(rsize == 0) return 0;

    *resRect = rects;

    if(flag == 1){
        assert(manager->aligner != NULL);

        for(int i = 0; i < rsize; i++){
            int cx = rects[i].x + (rects[i].width  >> 1);
            int cy = rects[i].y + (rects[i].height >> 1);

            int faceSize = rects[i].width / OBJECT_FACTOR;

            rects[i].x = cx - (faceSize >> 1);
            rects[i].y = cy - (faceSize >> 1);
            rects[i].width = faceSize;
            rects[i].height = faceSize;
        }

        delete [] confs;

        return rsize;
    }

    assert(manager->tracker != NULL);

    Shape meanShape = manager->tracker->meanShape;


    if(MAX_FACE_SIZE == 1){
        HRect rect = rects[0];
        Shape shape = meanShape;
        float maxScore = confs[0];

        for(int i = 1; i < rsize; i++){
            if(confs[i] > maxScore){
                rect = rects[i];
                maxScore = confs[i];
            }
        }

        int cx = rect.x + (rect.width >> 1);
        int cy = rect.y + (rect.height >> 1);

        float scale = (float)rect.width / ((float)SAMPLE_IMAGE_SIZE * OBJECT_FACTOR);
        int ptsSize = shape.ptsSize;

        for(int p = 0; p < ptsSize; p++){
            shape.pts[p] = shape.pts[p] * scale + cx;
            shape.pts[p + ptsSize] = shape.pts[p + ptsSize] * scale + cy;
        }

        manager->lastShapes[0] = shape;
        manager->lastSize = 1;
    }
    else {
        HRect lastRects[MAX_FACE_SIZE];

        for(int i = 0; i < manager->lastSize; i++){
            HRect rect = get_shape_rect(manager->lastShapes[i]);
            int cx = rect.x + (rect.width >> 1);
            int cy = rect.y + (rect.height >> 1);
            int faceSize = rect.width * OBJECT_FACTOR;

            rect.x = cx - (faceSize >> 1);
            rect.y = cy - (faceSize >> 1);
            rect.width = rect.height = faceSize;
            lastRects[i] = rect;
        }

        for(int i = 0; i < manager->lastSize; i++){
            int area2 = rects[i].width * rects[i].height * 0.6f;
            for(int j = i + 1; j < manager->lastSize; j++) {
                int area = calc_overlapping_area(lastRects[i], lastRects[j]);

                if(area > area2){
                    HU_SWAP(manager->lastShapes[j], manager->lastShapes[manager->lastSize - 1], Shape);
                    HU_SWAP(lastRects[j], lastRects[manager->lastSize - 1], HRect);
                    manager->lastSize--;
                    j --;
                }
            }
        }

        //remove
        for(int i = 0; i < rsize; i++){
            int area2 = rects[i].width * rects[i].height * 0.6f;
            for(int j = 0; j < manager->lastSize; j++){
                int area = calc_overlapping_area(rects[i], lastRects[j]);

                if(area > area2){
                    HU_SWAP(rects[i], rects[rsize - 1], HRect);
                    HU_SWAP(confs[i], confs[rsize - 1], float);
                    rsize --;
                    i--;
                    break;
                }
            }
        }

        //sort
        for(int i = 0; i < rsize; i++){
            float maxScore = confs[i];
            int id = i;
            for(int j = i + 1; j < rsize; j++){
                if(confs[j] > maxScore){
                    maxScore = confs[j];
                    id = j;
                }
            }

            if(i != id){
                HU_SWAP(rects[i], rects[id], HRect);
                HU_SWAP(confs[i], confs[id], float);
            }
        }

        rsize = HU_MIN(rsize + manager->lastSize, MAX_FACE_SIZE) - manager->lastSize;

        //push back
        for(int i = 0; i < rsize; i++){
            HRect rect = rects[i];
            Shape shape = meanShape;
            int ptsSize = shape.ptsSize;

            int cx = rect.x + (rect.width >> 1);
            int cy = rect.y + (rect.height >> 1);

            float scale = (float)rects[i].width / ((float)SAMPLE_IMAGE_SIZE * OBJECT_FACTOR);

            for(int p = 0; p < ptsSize; p++){
                shape.pts[p] = shape.pts[p] * scale + cx;
                shape.pts[p + ptsSize] = shape.pts[p + ptsSize] * scale + cy;
            }

            manager->lastShapes[manager->lastSize++] = shape;
        }
    }

    delete [] confs;


    return manager->lastSize;
}


int align_face(FaceManager *manager, uint8_t *img, int width, int height, int stride, Shape **shapes){

    if(manager == NULL || manager->aligner == NULL || img == NULL)
        return 0;

    int rsize;
    HRect *rects;

    init_detect_factor(manager->detector, 0.1, 1.0, 0.1, 15);
    rsize = detect_face(manager, img, width, height, stride, 1, &rects);

    if(rsize == 0) return 0;

    *shapes = new Shape[rsize];

    for(int i = 0; i < rsize; i++){
        HRect rect = rects[i];
        Shape shape;

        predict(manager->aligner, img, width, height, stride, rect, shape);

        (*shapes)[i] = shape;
    }

    delete  [] rects;

    return rsize;
}


int validate_face(FaceManager *manager, uint8_t *img, int width, int height, int stride, Shape &shape){
    TranArgs arg;
    float scale;

    uint8_t patch[100 * 100];
    HRect rect;
    HPoint2f cenS, cenD;

    int WINW = manager->detector->WINW;
    int WINH = manager->detector->WINH;
    int ret;

    similarity_transform(shape, manager->tracker->meanShape, arg);

    rect = get_shape_rect(shape);

    scale = WINW / (rect.width * OBJECT_FACTOR);

    cenS.x = rect.x + (rect.width >> 1);
    cenS.y = rect.y + (rect.height >> 1);

    cenD.x = WINW >> 1;
    cenD.y = WINH >> 1;

    affine_image(img, width, height, stride, cenS, patch, WINW, WINH, WINW, cenD, scale, arg.angle);

    ret = predict(manager->detector, patch, WINW, WINH, WINW);

    /*
    cv::Mat sImg(WINW, WINH, CV_8UC1, patch);
    cv::imshow("sImg", sImg);
    cv::waitKey();
    //*/

    return ret;
}


int track_one_face(FaceManager *manager, uint8_t *img, int width, int height, int stride, Shape &resShape){

    int rsize = 0;
    static int INTER = 0;

    if(manager->tflag == 1){
        HRect *rects;

        init_detect_factor(manager->detector, 0.2, 1.0, 0.1, 10);
        rsize = detect_face(manager, img, width, height, stride, 0, &rects);
        if(rsize == 0) return 0;

        //show_rect(img, width, height, stride, rects[0], "detect");
        assert(rsize <= MAX_FACE_SIZE);

        rsize = 1;

        manager->tflag = 0;
        manager->lastSize = 1;

        delete [] rects;
    }

    Shape shape = manager->lastShapes[0];

    //show_shape(img, width, height, stride, shape, "cur");
    predict(manager->tracker, img, width, height, stride, shape);
    //show_shape(img, width, height, stride, shape, "res");

    if((++INTER) % 5 == 0 && manager->tflag != 1){
        manager->tflag = !validate_face(manager, img, width, height, stride, shape);
        if(manager->tflag) shape.ptsSize = 0;
    }

    manager->lastShapes[0] = shape;

    resShape = shape;

    return 1;
}


int track_mul_face(FaceManager *manager, uint8_t *img, int width, int height, int stride, Shape *resShapes){

    int rsize = 0;
    static int INTER = 0;

    if(INTER % 25 == 0){
        HRect *rects;

        init_detect_factor(manager->detector, 0.2, 1.0, 0.1, 12);
        rsize = detect_face(manager, img, width, height, stride, 0, &rects);
        if(rsize == 0) return 0;
    }

    INTER++;
    int flags[MAX_FACE_SIZE + 2];
    memset(flags, 0, sizeof(int) * (MAX_FACE_SIZE + 2));

    for(int i = 0; i < manager->lastSize; i++){
        Shape shape = manager->lastShapes[i];
        predict(manager->tracker, img, width, height, stride,  shape);

        if(INTER % 5 == 0 && validate_face(manager, img, width, height, stride, shape) == 0){
            INTER = 0;
            flags[i] = 1;
            continue;
        }

        manager->lastShapes[i] = shape;
    }

    int count = 0;

    for(int i = 0; i < manager->lastSize; i++){
        if(flags[i] == 0){
            resShapes[count] = manager->lastShapes[i];
            count++;
        }
    }

    if(manager->lastSize != count){
        for(int i = 0; i < count; i++)
            manager->lastShapes[i] = resShapes[i];
        manager->lastSize = count;
    }

    return manager->lastSize;
}


void release_data(FaceManager *manager){
    if(manager == NULL)
        return ;

    release(&manager->aligner);
    release(&manager->tracker);
    release(&manager->detector);

    manager->lastSize = 0;

    return ;
}


void release(FaceManager **manager){
    if(*manager == NULL )
        return;

    delete *manager;
    *manager = NULL;
}


FaceManager *gManager = NULL;

uint8_t *gImgBuffer = NULL, *gRotateBuf = NULL;
int gCapacity = 0, gCapRotate = 0;


int load_models(const char *detModel, const char *alignModel, const char *trackModel){
    if(gImgBuffer != NULL)
        delete [] gImgBuffer;

    gImgBuffer = NULL;
    gCapacity = 0;

    if(gRotateBuf != NULL)
        delete [] gRotateBuf;

    gRotateBuf = NULL;
    gCapRotate = 0;

    if(gManager != NULL)
        delete gManager;

    return load(detModel, alignModel, trackModel, &gManager);
}


void release_models(){
    release(&gManager);

    if(gImgBuffer != NULL)
        delete [] gImgBuffer;
    gImgBuffer = NULL;

    if(gRotateBuf != NULL)
        delete [] gRotateBuf;
    gRotateBuf = NULL;

    gManager = NULL;
    gCapacity = 0;
    gCapRotate = 0;
}


static void rotate_points(float* pts, int ptsSize, int width, int height, int rflag){
    if(rflag == ROATE_FLAG_0)
        return;

    else if(rflag == ROATE_FLAG_90){
        for(int i = 0; i < ptsSize; i ++){
            int j = i << 1;
            float x = pts[j + 1];
            float y = width - pts[j] - 1;

            pts[j] = x;
            pts[j + 1] = y;
        }

    }
    else if(rflag == ROATE_FLAG_180){
        for(int i = 0; i < ptsSize; i++){
            int j = i << 1;
            float x = width  - 1 - pts[j];
            float y = height - 1 - pts[j + 1];

            pts[j] = x;
            pts[j + 1] = y;

        }
    }
    else if(rflag == ROATE_FLAG_270){
        for(int i = 0; i < ptsSize; i++){
            int j = i << 1;
            float x = height - 1 - pts[j + 1];
            float y = pts[j];

            pts[j] = x;
            pts[j + 1] = y;
        }
    }
}



int process_track(uint8_t *img, int width, int height, int stride, int format, int rflag, float *resPts){
    assert(resPts != NULL);

    if(gManager == NULL || gManager->tracker == NULL) return 0;

    uint8_t *imgData;
    int cols, rows, step;

    //control image format
    if(format == FT_IMAGE_BGRA){
        if(width * height > gCapacity){
            if(gImgBuffer != NULL)
                delete [] gImgBuffer;

            gCapacity = width * height;
            gImgBuffer = new uint8_t[gCapacity];
        }

        bgra2gray(img, width, height, stride, gImgBuffer);

        imgData = gImgBuffer;
        cols = width;
        rows = height;
        step = width;
    }
    else if(format == FT_IMAGE_NV21){
        imgData = img;
        cols = width;
        rows = height;
        step = stride;
    }

    //control image rotation
    if(rflag != 0){
        if(gCapRotate < step * rows){
            if(gRotateBuf != NULL)
                delete [] gRotateBuf;

            gCapRotate = step * rows;
            gRotateBuf = new uint8_t[gCapRotate];
        }

        rotate_width_degree(imgData, cols, rows, step, gRotateBuf, width, height, stride, rflag);

        imgData = gRotateBuf;

        cols = width;
        rows = height;
        step = stride;
    }

    Shape shapes[MAX_FACE_SIZE];
    int rsize = track_mul_face(gManager, imgData, cols, rows, step, shapes);

    if(rsize > 0){
        float *ptrS;
        float *ptrD = resPts;
        int ptsSize;
        for(int i = 0; i < rsize; i++){
            ptrS = shapes[i].pts;

            ptsSize = shapes[i].ptsSize;

            for(int p = 0; p < ptsSize; p++){
                int id = p << 1;

                ptrD[id] = ptrS[p];
                ptrD[id + 1] = ptrS[p + ptsSize];
            }

            rotate_points(ptrD, ptsSize, cols, rows, rflag);

            ptrD += ptsSize * 2;
        }
    }

    return rsize;
}


int process_align(uint8_t *img, int width, int height, int stride, int format, float **resShape){
    int x0 = width * 0.2f;
    int y0 = height * 0.2f;

    int gWidth  = width * 1.4f;
    int gHeight = height * 1.4f;
    int gStride = gWidth;

    uint8_t *grayBuffer = new uint8_t[gWidth * gHeight];
    uint8_t *gray = grayBuffer + y0 * gStride + x0;

    memset(grayBuffer, 0, sizeof(uint8_t) * gWidth * gHeight);

    if(format == FT_IMAGE_NV21){
        for(int y = 0; y < height; y++)
            memcpy(gray + y * gStride, img + y * stride, sizeof(uint8_t) * width);
    }

    else if(format == FT_IMAGE_BGRA){
        uint8_t *buffer = new uint8_t[width * height];

        bgra2gray(img, width, height, stride, buffer);
        stride = width;

        for(int y = 0; y < height; y++)
            memcpy(gray + y * gStride, buffer + y * stride, sizeof(uint8_t) * width);

        delete [] buffer;
    }

    Shape *shapes;
    int rsize = align_face(gManager, grayBuffer, gWidth, gHeight, gStride, &shapes);

    if(rsize > 0){
        int ptsSize = shapes[0].ptsSize;

        *resShape = new float[ptsSize * 2 * rsize];

        float *ptrS;
        float *ptrD = *resShape;
        for(int i = 0; i < rsize; i++){
            ptrS = shapes[i].pts;

            for(int p = 0; p < ptsSize; p++){
                ptrD[p << 1] = ptrS[p] - x0;
                ptrD[(p << 1) + 1] = ptrS[p + ptsSize] - y0;
            }

            ptrD += ptsSize * 2;
        }

        delete [] shapes;
    }

    delete [] grayBuffer;

    return rsize;
}


#define CEN_AREA_X0 -21.594006f
#define CEN_AREA_Y0 -13.702213f
#define CEN_AREA_X1  24.395615f
#define CEN_AREA_Y1  26.736389f

void estimate_angle(Shape &meanShape, Shape &shape, float &ex, float &ey, float &ez, float &scale){
    float cx, cy;
    TranArgs arg;
    int ptsSize = shape.ptsSize;

    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;

    similarity_transform(shape, meanShape, arg);

    ez = -arg.angle;
    scale = arg.scale;

    arg.cen2 = arg.cen1;

    affine_shape(shape, shape, arg);

    cx = 0.0f;
    cy = 0.0f;

    for(int i = 0; i < ptsSize; i++){
        float x = shape.pts[i];
        float y = shape.pts[i + ptsSize];

        cx += x;
        cy += y;

        minx = HU_MIN(minx, x);
        maxx = HU_MAX(maxx, x);

        miny = HU_MIN(miny, y);
        maxy = HU_MAX(maxy, y);
    }

    cx /= ptsSize;
    cy /= ptsSize;

    cx -= (maxx + minx) / 2;

    if(ptsSize == 68)
        cy = shape.pts[33 + ptsSize] - (miny + maxy) / 2;
    else
        cy = shape.pts[100 + ptsSize] - (miny + maxy) / 2;

    float w = (CEN_AREA_X1 - CEN_AREA_X0) / 2;
    float h = (CEN_AREA_Y1 - CEN_AREA_Y1) / 2;

    ey = asinf(cx / w);
    ex = asinf(cy / h);
}


void get_angles(float *oriShape, int ptsSize, float &ex, float &ey, float &ez){
    float scale;

    Shape shape;

    shape.ptsSize = ptsSize;
    for(int i = 0; i < ptsSize; i++){
        int j = i << 1;

        shape.pts[i] = oriShape[j];
        shape.pts[i + ptsSize] = oriShape[j + 1];
    }

    estimate_angle(gManager->aligner->meanShape, shape, ex, ey, ez, scale);
}

