#include "cascade.h"


void validate(JDADetector *detector, SampleSet *set, int times);


void init_train_params(TrainParams *params, NegGenerator *generator, int WINW, int WINH, int i){
    params->depth = 4;
    params->treeSize = 272;
    params->flag = 1;
    params->radius = 0.4;
    params->npRate = 1;

    generator->tflag = 1;
    generator->dx = 0.1 * WINW;
    generator->dy = 0.1 * WINH;
    generator->maxCount = 20;

    switch(i){
        case 0:
            params->recall = 1.0f;
            params->prob = 0.9f;
            params->npRate = 5;

            generator->dx = WINW;
            generator->dy = WINH;
            generator->tflag = 0;
            generator->maxCount = 10;

            break;

        case 1:
            params->recall = 1.0f;
            params->prob = 0.8f;

            generator->dx = 0.3 * WINW;
            generator->dy = 0.3 * WINH;

            break;

        case 2:
            params->recall = 1.0f;
            params->prob = 0.7f;
            params->flag = 0;

            break;

        case 3:
            params->recall = 1.0f;
            params->prob = 0.6f;
            params->flag = 0;

            break;

        case 4:
            params->recall = 1.0f;
            params->prob = 0.5f;
            params->flag = 0;
            params->npRate = 0.3f;

            break;

        default:
            params->recall = 1.0f;
            params->prob = 0.9f;
            params->npRate = 1;
            params->flag = 0;

            break;
    }
}


int train(JDADetector **rdetector, const char *posFile, const char *negFile, int WINW, int WINH, int stage){

    SampleSet *posSet = NULL, *negSet = NULL, *oriSet = NULL;

    JDADetector *detector = NULL;
    NegGenerator *generator = NULL;

    TrainParams params;
    int ret;


    if(load(POS_SET_FILE, &oriSet) != 0){
        Shape meanShape;

        ret = read_mean_shape(posFile, meanShape, HU_MAX(WINW, WINH));
        if(ret != 0) return 1;

        ret = read_positive_samples(posFile, meanShape, WINW << 1, WINH << 1, 0, &oriSet);
        if(ret != 0) return 1;

        write_images(oriSet, "log/pos", oriSet->ssize / 4000 + 1);

        save(POS_SET_FILE, oriSet);
    }

    posSet = new SampleSet;
    memset(posSet, 0, sizeof(SampleSet));

    posSet->meanShape = oriSet->meanShape;
    posSet->WINW = WINW;
    posSet->WINH = WINH;

    printf("GENERATE TRANSFORM POSITIVE SAMPLES\n");
    generate_transform_samples(oriSet, 1, posSet);
    release(&oriSet);

    write_images(posSet, "log/pos", posSet->ssize / 4000 + 1);

    negSet = new SampleSet;
    memset(negSet, 0, sizeof(SampleSet));

    negSet->meanShape = posSet->meanShape;
    negSet->WINW = posSet->WINW;
    negSet->WINH = posSet->WINH;
    negSet->ptsSize = posSet->ptsSize;

    for(int i = stage - 1; i >= 0; i--){
        char filePath[256];
        sprintf(filePath, "cascade_%d.dat", i);

        if(load(filePath, &detector) == 0){
            assert(detector->WINW == posSet->WINW &&
                    detector->WINH == posSet->WINH);

            validate(detector, posSet, 1000);
            break;
        }
    }

    if(detector == NULL){
        detector = new JDADetector;
        detector->meanShape = posSet->meanShape;
        detector->WINW = posSet->WINW;
        detector->WINH = posSet->WINH;
        detector->forests = new Forest[stage];

        memset(detector->forests, 0, sizeof(Forest) * stage);

        detector->ssize = 0;
        detector->capacity = stage;
    }
    else{
        Forest *forests = new Forest[stage];

        memset(forests, 0, sizeof(Forest) * stage);

        if(detector->ssize > 0){
            memcpy(forests, detector->forests, sizeof(Forest) * detector->ssize);
            delete [] detector->forests;
        }

        detector->forests = forests;
        detector->capacity = stage;
    }

    if(detector->ssize == 0){
        char command[256];
        sprintf(command, "find neg_patches -name \"*.bmp\" > %s", NEG_PATCH_LIST_FILE);
        sprintf(command, "find neg_patches -name \"*.jpg\" >> %s", NEG_PATCH_LIST_FILE);
        ret = system(command);
        read_negative_patch_samples(NEG_PATCH_LIST_FILE, negSet);
    }

    generate_negative_images(negFile, NEG_IMAGES_FILE);

    print_info(posSet);
    print_info(negSet);

    generator = new NegGenerator;
    memset(generator, 0, sizeof(NegGenerator));

    generator->fin = fopen(NEG_IMAGES_FILE, "rb"); assert(generator->fin != NULL);
    ret = fread(&generator->isize, sizeof(int), 1, generator->fin);assert(ret == 1);

    printf("NEGATIVE IMAGE SIZE: %d\n", generator->isize);

    generator->id = 0;
    generator->forests = detector->forests;
    generator->meanShape = detector->meanShape;

    strcpy(generator->filePath, NEG_IMAGES_FILE);

    ret = system("mkdir -p model");


    for(int i = detector->ssize; i < stage; i++){
        printf("-------------------- Train Cascade %d --------------------\n", i);
        char outfile[256];
        generator->fsize = i + 1;

        init_train_params(&params, generator, posSet->WINW, posSet->WINH, i);

        printf("RECALL = %f PROB = %f\n", params.recall, params.prob);
        train(detector->forests + i, posSet, negSet, &params, generator);

        detector->ssize++;

        sprintf(outfile, "model/cascade_%d.dat", i);
        printf("SAVE MODEL: %s\n", outfile);
        save(outfile, detector);

        validate(detector, posSet, 1000);

        //write_images(posSet, "log/pos", posSet->ssize / 4000 + 1);
        printf("----------------------------------------------------------\n");
    }

    *rdetector = detector;

    release(&posSet);
    release(&negSet);
    release(&generator);

    return 0;
}


int predict(JDADetector *detector, uint8_t *img, int stride, Shape &shape, float &score){
#if 1
    if(predict_one(detector->forests, detector->meanShape, img, stride, shape, score) == 0)
        return 0;

    for(int i = 1; i < detector->ssize; i++)
        if(predict(detector->forests + i, detector->meanShape, img, stride, shape, score) == 0)
            return 0;
#else
    for(int i = 0; i < detector->ssize; i++)
        if(predict(detector->forests + i, detector->meanShape, img, stride, shape, score) == 0)
            return 0;

#endif
    return 1;
}


void validate(JDADetector *detector, SampleSet *set, int times){
    Shape meanShape = detector->meanShape;

    for(int i = 0; i < set->ssize; i++){
        Sample *sample = set->samples[i];

        int ret = 0;

        for(int t = 0; t < times; t++){
            argument(meanShape, sample->curShape);
            ret = predict(detector->forests, detector->ssize, detector->meanShape, sample->img, sample->stride, sample->curShape, sample->score);
            if(ret == 1) break;
        }

        if(ret == 0){
            HU_SWAP(set->samples[i], set->samples[set->ssize - 1], Sample*);
            release(set->samples + set->ssize - 1);
            i--;
            set->ssize --;
        }
    }
}


int load(const char *filePath, JDADetector **rdetector){
    FILE *fin = fopen(filePath, "rb");
    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    int ret;
    int ptsSize;

    JDADetector *detector = new JDADetector;

    ret = fread(&detector->meanShape, sizeof(Shape), 1, fin); assert(ret == 1);
    ret = fread(&detector->WINW, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&detector->WINH, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&detector->ssize, sizeof(int), 1, fin); assert(ret == 1);

    detector->forests = new Forest[detector->ssize];
    detector->capacity = detector->ssize;

    memset(detector->forests, 0, sizeof(Forest) *detector->ssize);

    ptsSize = detector->meanShape.ptsSize;

    assert(ptsSize == PTS_SIZE);

    for(int i = 0; i < detector->ssize; i++)
        load(fin, ptsSize, detector->forests + i);

    fclose(fin);

    set_detect_factor(detector, 0.2, 0.9, 0.2, 0.1, 10);

    *rdetector = detector;

    return 0;
}


int save(const char *filePath, JDADetector *detector){
    FILE *fout = fopen(filePath, "wb");
    if(fout == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    int ret;
    int ptsSize = detector->meanShape.ptsSize;

    ret = fwrite(&detector->meanShape, sizeof(Shape), 1, fout); assert(ret == 1);
    ret = fwrite(&detector->WINW, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&detector->WINH, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&detector->ssize, sizeof(int), 1, fout); assert(ret == 1);

    for(int i = 0; i < detector->ssize; i++)
        save(fout, ptsSize, detector->forests + i);


    fclose(fout);

    return 0;
}


int calc_overlapping_area(HRect &rect1, HRect &rect2){
    int cx1 = rect1.x + rect1.width / 2;
    int cy1 = rect1.y + rect1.height / 2;
    int cx2 = rect2.x + rect2.width / 2;
    int cy2 = rect2.y + rect2.height / 2;

    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;

    if(abs(cx1 - cx2) < rect1.width / 2 + rect2.width/2 && abs(cy1 - cy2) < rect1.height / 2 + rect2.height / 2){
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


int merge_rects(HRect *rects, Shape *shapes, float *confs, int size){
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
        shapes[i] = shapes[size - 1];

        i --;
        size --;
    }

    delete []flags;
    flags = NULL;

    return size;
}


void set_detect_factor(JDADetector *detector, float sImgScale, float eImgScale, float sOffScale, float eOffScale, int layer)
{
    if(sImgScale < eImgScale){
        if(sOffScale < eOffScale)
            HU_SWAP(sOffScale, eOffScale, float);
    }
    else {
        if(sOffScale > eOffScale)
            HU_SWAP(sOffScale, eOffScale, float);
    }

    detector->sImgScale = sImgScale;
    detector->eImgScale = eImgScale;
    detector->sOffScale = sOffScale;
    detector->eOffScale = eOffScale;

    detector->layer = layer;
}


int detect_one_scale(JDADetector *cc, float scale, float offScale,
                     uint8_t *iImg, int width, int height, int stride,
                     HRect *resRect, Shape *resShape, float *resScores){
    int WINW = cc->WINW;
    int WINH = cc->WINH;

    int dx = WINW * offScale;
    int dy = WINH * offScale;

    int count = 0;

    int ptsSize = cc->meanShape.ptsSize;

    for(int y = 0; y <= height - WINH; y += dy){
        for(int x = 0; x <= width - WINW; x += dx){
            uint8_t *ptr = iImg + y * stride + x;

            Shape shape = cc->meanShape;
            float score;

            if(predict(cc, ptr, stride, shape, score) == 1){
                /*
                printf("score: %f\n", score);
                show_shape(ptr, WINW, WINH, stride, shape, "shape");
                //*/
                //
                resRect[count].x = x * scale;
                resRect[count].y = y * scale;

                resRect[count].width  = WINW * scale;
                resRect[count].height = WINH * scale;

                resScores[count] = score;

                for(int p = 0; p < ptsSize; p++){
                    resShape[count].pts[p].x = (shape.pts[p].x + x) * scale;
                    resShape[count].pts[p].y = (shape.pts[p].y + y) * scale;
                }

                resShape[count].ptsSize = ptsSize;

                count++;
            }
        }
    }

    count = merge_rects(resRect, resShape, resScores, count);

    return count;
}


int calculate_max_size(int width, int height, float scale, int winSize){
    int minwh = HU_MIN(width, height);

    assert(scale < 1.0f);

    int size = minwh * scale;
    float scale2 = (float)winSize / size;

    width ++;
    height ++;

    if(scale2 < 1)
        return width * height;

    return (width * scale2 + 0.5f) * (height * scale2 + 0.5f);
}


HRect get_face_rect(Shape *shape, int width, int height){
    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;

    for(int i = 0; i < shape->ptsSize; i++){
        minx = HU_MIN(minx, shape->pts[i].x);
        maxx = HU_MAX(maxx, shape->pts[i].x);
        miny = HU_MIN(miny, shape->pts[i].y);
        maxy = HU_MAX(maxy, shape->pts[i].y);
    }

    float cx = (maxx + minx) * 0.5f;
    float cy = (maxy + miny) * 0.5f;

    float faceSize = HU_MAX(maxx - minx + 1, maxy - miny + 1) * SCALE_FACTOR;
    float len = 0;

    HRect rect;

    rect.x = cx - faceSize * 0.5f;
    rect.y = cy - faceSize * 0.5f;
    rect.x = HU_MAX(0, rect.x);
    rect.y = HU_MAX(0, rect.y);

    rect.width  = rect.x + faceSize - 1;
    rect.height = rect.y + faceSize - 1;

    if(rect.width >= width){
        rect.width = faceSize;
        rect.x = width - faceSize;
    }
    else {
        rect.width = rect.width - rect.x + 1;
    }

    if(rect.height > height){
        rect.height = faceSize;
        rect.y = height - faceSize;
    }
    else {
        rect.height = rect.height - rect.y + 1;
    }

    return rect;
}

int detect(JDADetector *cc, uint8_t *img, int width, int height, int stride, HRect **resRect, Shape **resShapes, float **resScores){
    int WINW, WINH, capacity;
    float imgScale, imgFactor, offScale, offFactor;

    uint8_t *dImg, *ptrSrc, *ptrDst;

    HRect *rects = NULL;
    float *scores = NULL;
    Shape *shapes = NULL;
    int top = 0;

    int srcw, srch, srcs, dstw, dsth, dsts;
    int minSide;
    int count;

    WINW = cc->WINW;
    WINH = cc->WINH;

    imgScale = cc->sImgScale;
    offScale = cc->sOffScale;

    imgFactor = powf(cc->eImgScale / cc->sImgScale, 1.0f / (cc->layer - 1));
    offFactor = powf(cc->eOffScale / cc->sOffScale, 1.0f / (cc->layer - 1));

    capacity = calculate_max_size(width, height, imgScale, HU_MAX(WINW, WINH));

    dImg = new uint8_t[capacity * 2]; assert(dImg != NULL);

    const int BUFFER_SIZE = 1000;

    rects  = new HRect[BUFFER_SIZE]; assert(rects != NULL);
    scores = new float[BUFFER_SIZE]; assert(scores != NULL);
    shapes = new Shape[BUFFER_SIZE]; assert(shapes != NULL);

    memset(rects, 0, sizeof(HRect)  * BUFFER_SIZE);
    memset(scores, 0, sizeof(float) * BUFFER_SIZE);
    memset(shapes, 0, sizeof(Shape) * BUFFER_SIZE);

    ptrSrc = img;
    ptrDst = dImg;

    srcw = width;
    srch = height;
    srcs = stride;

    count = 0;

    minSide = HU_MIN(width, height);

    for(int i = 0; i < cc->layer; i++){
        uint8_t *buffer;
        float scale2 = HU_MIN(WINW, WINH) / (minSide * imgScale);

        dstw = scale2 * width;
        dsth = scale2 * height;
        dsts = dstw;

        resizer_bilinear_gray(ptrSrc, srcw, srch, srcs, ptrDst, dstw, dsth, dsts);

        //buffer = mean_filter_3x3_res(ptrDst, dstw, dsth, dsts);
        count += detect_one_scale(cc, 1.0f / scale2, offScale, ptrDst, dstw, dsth, dsts, rects + count, shapes + count, scores + count);
        //delete [] buffer;

        ptrSrc = ptrDst;

        srcw = dstw;
        srch = dsth;
        srcs = dsts;

        if(ptrDst == dImg)
            ptrDst = dImg + dstw * dsth;
        else
            ptrDst = dImg;

        imgScale *= imgFactor;
        offScale *= offFactor;
    }

    count = merge_rects(rects, shapes, scores, count);

    if(count > 0){
        //*
        for(int i = 0; i < count; i++)
            rects[i] = get_face_rect(shapes + i, width, height);
        // */

        *resRect = rects;
        *resShapes = shapes;
        *resScores = scores;
    }
    else {
        delete [] rects;
        delete [] shapes;
        delete [] scores;
    }

    delete [] dImg;

    return count;
}


void release_data(JDADetector *detector){
    if(detector != NULL){
        for(int i = 0; i < detector->ssize; i++)
            release_data(detector->forests + i);

        delete [] detector->forests;
    }

    detector->forests = NULL;
    detector->capacity = 0;
    detector->ssize = 0;
}


void release(JDADetector **detector){
    if(*detector != NULL){
        release_data(*detector);
        delete *detector;
    }

    *detector = NULL;
}
