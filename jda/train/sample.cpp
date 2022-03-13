#include "sample.h"


void affine_sample(uint8_t *img, int width, int height, int stride, Shape &shape, float scale, float angle, uint8_t *dst);

int read_pts_file(const char *filePath, Shape *shape)
{
    FILE *fin = fopen(filePath, "r");

    char line[256];
    float buffer[202];
    int ptsSize;
    char *ret;

    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 0;
    }

    ret = fgets(line, 255, fin);
    ret = fgets(line, 255, fin);

    sscanf(line, "n_points:  %d", &ptsSize);

    ret = fgets(line, 255, fin);

    for(int i = 0; i < ptsSize; i++){
        if(fgets(line, 255, fin) == NULL) {
            fclose(fin);
            printf("END of FILE: %s\n", filePath);
            return 0;
        }

        int ret = sscanf(line, "%f %f\n", buffer + i * 2, buffer + i * 2 + 1);

        if(ret == 0) break;
    }

    fclose(fin);

    shape->ptsSize = ptsSize;
    assert(shape->ptsSize <= MAX_PTS_SIZE);

    for(int i = 0; i < shape->ptsSize; i++){
        shape->pts[i].x = buffer[i * 2];
        shape->pts[i].y = buffer[i * 2 + 1];
    }

    return shape->ptsSize;
}


int write_pts_file(const char *filePath, float *shape, int ptsSize)
{
    FILE *fout = fopen(filePath, "w");

    char line[256];

    if(fout == NULL)
    {
        printf("Can't open file %s\n", filePath);
        return 1;
    }


    fprintf(fout, "version: 1\n");
    fprintf(fout, "n_points:  %d\n", ptsSize);
    fprintf(fout, "{\n");

    for(int i = 0; i < ptsSize; i++)
        fprintf(fout, "%f %f\n", shape[i << 1], shape[(i << 1) + 1]);

    fprintf(fout, "}");
    fclose(fout);

    return 0;
}


int write_pts_file(const char *filePath, Shape *shape)
{
    FILE *fout = fopen(filePath, "w");

    if(fout == NULL){
        printf("Can't open file %s\n", filePath);
        return 0;
    }

    char *ret;
    fprintf(fout, "version: 1\n");;
    fprintf(fout, "n_points: %d\n", shape->ptsSize);
    fprintf(fout, "{\n");

    for(int i = 0; i < shape->ptsSize; i ++)
        fprintf(fout, "%f %f\n", shape->pts[i].x, shape->pts[i].y);

    fprintf(fout, "}");
    fclose(fout);

    return shape->ptsSize;
}


void calculate_mean_shape_global(Shape *shapes, int size, int ptsSize, int SHAPE_SIZE, Shape &meanShape)
{
    double cxs[MAX_PTS_SIZE], cys[MAX_PTS_SIZE];

    float minx, miny, maxx, maxy;
    float cx, cy, w, h, faceSize, scale;

    assert(ptsSize = PTS_SIZE);

    memset(cxs, 0, sizeof(double) * MAX_PTS_SIZE);
    memset(cys, 0, sizeof(double) * MAX_PTS_SIZE);

    for(int i = 0; i < size; i++){
        Shape &shape = shapes[i];

        minx = FLT_MAX; maxx = -FLT_MAX;
        miny = FLT_MAX; maxy = -FLT_MAX;

        for(int j = 0; j < ptsSize; j++){
            float x = shape.pts[j].x;
            float y = shape.pts[j].y;

            minx = HU_MIN(x, minx);
            maxx = HU_MAX(x, maxx);
            miny = HU_MIN(y, miny);
            maxy = HU_MAX(y, maxy);
        }

        w = maxx - minx + 1;
        h = maxy - miny + 1;

        cx = (maxx + minx) / 2;
        cy = (maxy + miny) / 2;

        faceSize = HU_MAX(w, h);

        scale = SHAPE_SIZE / faceSize;

        for(int j = 0; j < ptsSize; j++){
            cxs[j] += (shape.pts[j].x - cx) * scale + SHAPE_SIZE / 2;
            cys[j] += (shape.pts[j].y - cy) * scale + SHAPE_SIZE / 2;
        }
    }

    minx =  FLT_MAX, miny =  FLT_MAX;
    maxx = -FLT_MAX, maxy = -FLT_MAX;

    meanShape.ptsSize = ptsSize;

    for(int j = 0; j < ptsSize; j++){
        meanShape.pts[j].x = cxs[j] / size;
        meanShape.pts[j].y = cys[j] / size;

        minx = HU_MIN(minx, meanShape.pts[j].x);
        maxx = HU_MAX(maxx, meanShape.pts[j].x);

        miny = HU_MIN(miny, meanShape.pts[j].y);
        maxy = HU_MAX(maxy, meanShape.pts[j].y);
    }

    w = maxx - minx + 1;
    h = maxy - miny + 1;

    cx = (maxx + minx) * 0.5f;
    cy = (maxy + miny) * 0.5f;

    faceSize = HU_MAX(w, h) * SCALE_FACTOR;

    scale = SHAPE_SIZE / faceSize;

    for(int j = 0; j < ptsSize; j++){
        meanShape.pts[j].x = (meanShape.pts[j].x - cx) * scale + (SHAPE_SIZE >> 1);
        meanShape.pts[j].y = (meanShape.pts[j].y - cy) * scale + (SHAPE_SIZE >> 1);
    }
}


void argument(Shape &shape, Shape &res)
{
    float shiftX, shiftY;
    float sina, cosa, maxX, maxY, minX, minY;
    float cx = 0.0f, cy = 0.0f;
    int ptsSize = shape.ptsSize;
    float scale, angle;

    static cv::RNG rng(cv::getTickCount());

    scale = rng.uniform(0.95f, 1.05f);
    angle = rng.uniform(-HU_PI / 36, HU_PI / 36);

    cosa = cos(angle);
    sina = sin(angle);

    float factor = 0.03f;
    shiftX = rng.uniform(-factor, factor);
    shiftY = rng.uniform(-factor, factor);

    minX = FLT_MAX; maxX = -FLT_MAX;
    minY = FLT_MAX; maxY = -FLT_MAX;

    cx = 0, cy = 0;

    for(int i = 0; i < ptsSize; i++){
        minX = HU_MIN(minX, shape.pts[i].x);
        maxX = HU_MAX(maxX, shape.pts[i].x);
        minY = HU_MIN(minY, shape.pts[i].y);
        maxY = HU_MAX(maxY, shape.pts[i].y);

        cx += shape.pts[i].x;
        cy += shape.pts[i].y;
    }

    cx /= ptsSize; cy /= ptsSize;

    shiftX = (maxX - minX + 1) * shiftX;
    shiftY = (maxY - minY + 1) * shiftY;

    for(int i = 0; i < ptsSize; i++){
        float x = (shape.pts[i].x - cx) / scale;
        float y = (shape.pts[i].y - cy) / scale;

        res.pts[i].x =  x * cosa + y * sina;
        res.pts[i].y = -x * sina + y * cosa;

        res.pts[i].x += cx;
        res.pts[i].y += cy;

        res.pts[i].x += shiftX;
        res.pts[i].y += shiftY;
    }

    res.ptsSize = shape.ptsSize;
}


void similarity_transform(Shape &src, Shape &dst, TranArgs &arg){
    int ptsSize = src.ptsSize;
    float cx1 = 0.0f, cy1 = 0.0f;
    float cx2 = 0.0f, cy2 = 0.0f;

    int i;

    for(i = 0; i <= ptsSize - 4; i += 4){
        cx1 += src.pts[i].x;
        cy1 += src.pts[i].y;
        cx1 += src.pts[i + 1].x;
        cy1 += src.pts[i + 1].y;
        cx1 += src.pts[i + 2].x;
        cy1 += src.pts[i + 2].y;
        cx1 += src.pts[i + 3].x;
        cy1 += src.pts[i + 3].y;
    }

    for(; i < ptsSize; i++){
        cx1 += src.pts[i].x;
        cy1 += src.pts[i].y;
    }

    cx1 /= ptsSize;
    cy1 /= ptsSize;

    for(i = 0; i <= ptsSize - 4; i += 4){
        cx2 += dst.pts[i].x;
        cy2 += dst.pts[i].y;
        cx2 += dst.pts[i + 1].x;
        cy2 += dst.pts[i + 1].y;
        cx2 += dst.pts[i + 2].x;
        cy2 += dst.pts[i + 2].y;
        cx2 += dst.pts[i + 3].x;
        cy2 += dst.pts[i + 3].y;
    }

    for(; i < ptsSize; i++){
        cx2 += dst.pts[i].x;
        cy2 += dst.pts[i].y;
    }

    cx2 /= ptsSize;
    cy2 /= ptsSize;

    float ssina = 0.0f, scosa = 0.0f, num = 0.0f;
    float var1 = 0.0f, var2 = 0.0f;

    for(int i = 0; i < ptsSize; i++){
        float sx = src.pts[i].x - cx1;
        float sy = src.pts[i].y - cy1;

        float dx = dst.pts[i].x - cx2;
        float dy = dst.pts[i].y - cy2;

        var1 += sqrtf(sx * sx + sy * sy);
        var2 += sqrtf(dx * dx + dy * dy);

        ssina += (sy * dx - sx * dy);
        scosa += (sx * dx + sy * dy);
    }

    num = 1.0f / sqrtf(ssina * ssina + scosa * scosa);

    arg.sina = ssina * num;
    arg.cosa = scosa * num;

    arg.scale = var2 / var1;

    arg.cen1.x = cx1;
    arg.cen1.y = cy1;
    arg.cen2.x = cx2;
    arg.cen2.y = cy2;

    arg.ssina = arg.scale * arg.sina;
    arg.scosa = arg.scale * arg.cosa;
}


void get_shape_rect(Shape &shape, HRect *rect){
    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;

    int cx, cy;
    int faceSize;

    for(int i = 0; i < shape.ptsSize; i++){
        minx = HU_MIN(minx, shape.pts[i].x);
        maxx = HU_MAX(maxx, shape.pts[i].x);

        miny = HU_MIN(miny, shape.pts[i].y);
        maxy = HU_MAX(maxy, shape.pts[i].y);
    }

    cx = 0.5f * (minx + maxx);
    cy = 0.5f * (miny + maxy);

    faceSize = HU_MAX(maxx - minx + 1, maxy - miny + 1);

    rect->x = cx - faceSize * 0.5f;
    rect->y = cy - faceSize * 0.5f;

    rect->width = rect->height = faceSize;
}


void extract_sample_from_image(uint8_t *img, int width, int height, int stride, Shape &meanShape, int WINW, int WINH, int border, Sample* sample){
    Shape curShape = sample->curShape;
    Shape oriShape = sample->oriShape;

    HRect rect;
    TranArgs arg;
    float scale;

    uint8_t *sImg;
    int swidth, sheight, sstride;

    float cx, cy;
    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;

    int x0, y0, x1, y1;
    int bl, br, bt, bb;
    int w, h;

    WINW += 2 * border;
    WINH += 2 * border;

    assert(oriShape.ptsSize == meanShape.ptsSize);

    //show_shape(img, width, height, stride, sample->oriShape, "ori");

    similarity_transform(oriShape, meanShape, arg);

    scale = arg.scale;

    swidth  = width * scale + 0.5f;
    sheight = height * scale + 0.5f;
    sstride = swidth;
    sImg = new uint8_t[sstride * sheight];

    resizer_bilinear_gray(img, width, height, stride, sImg, swidth, sheight, sstride);

    for(int i = 0; i < oriShape.ptsSize; i++){
        curShape.pts[i].x *= scale;
        curShape.pts[i].y *= scale;

        oriShape.pts[i].x *= scale;
        oriShape.pts[i].y *= scale;

        minx = HU_MIN(minx, oriShape.pts[i].x);
        maxx = HU_MAX(maxx, oriShape.pts[i].x);
        miny = HU_MIN(miny, oriShape.pts[i].y);
        maxy = HU_MAX(maxy, oriShape.pts[i].y);
    }

    cx = 0.5f * (minx + maxx);
    cy = 0.5f * (miny + maxy);

    x0 = cx - (WINW >> 1);
    y0 = cy - (WINH >> 1);

    x1 = x0 + WINW - 1;
    y1 = y0 + WINH - 1;

    bl = br = bt = bb = 0;

    if(x0 < 0) {
        bl = -x0;
        x0 = 0;
    }

    if(y0 < 0){
        bt = -y0;
        y0 = 0;
    }

    if(x1 > swidth - 1){
        br = x1 - swidth + 1;
        x1 = swidth - 1;
    }

    if(y1 > sheight - 1){
        bb = y1 - sheight + 1;
        y1 = sheight - 1;
    }

    if(sample->iImg == NULL)
        sample->iImg = new uint8_t[WINW * WINH];

    memset(sample->iImg, 0, sizeof(uint8_t) * WINW * WINH);

    w = WINW - bl - br;
    h = WINH - bt - bb;

    uint8_t *ptrDst = sample->iImg + bt * WINW + bl;
    uint8_t *ptrSrc = sImg + y0 * sstride + x0;

    for(int y = 0; y < h; y++)
        memcpy(ptrDst + y * WINW, ptrSrc + y * sstride, sizeof(uint8_t) * w);

    for(int i = 0; i < oriShape.ptsSize; i++){
        curShape.pts[i].x += (bl - x0);
        curShape.pts[i].y += (bt - y0);

        oriShape.pts[i].x += (bl - x0);
        oriShape.pts[i].y += (bt - y0);

        curShape.pts[i].x -= border;
        curShape.pts[i].y -= border;

        oriShape.pts[i].x -= border;
        oriShape.pts[i].y -= border;
    }

    sample->img = sample->iImg + border * WINW + border;
    sample->stride = WINW;

    sample->oriShape = oriShape;
    sample->curShape = curShape;

    //show_shape(sample->img, WINW - border * 2, WINH - border * 2, WINW, sample->oriShape, "orishape");
    //show_shape(sample->img, WINW - border * 2, WINH - border * 2, WINW, sample->curShape, "curshape");

    delete [] sImg;
}


int read_mean_shape(const char *listFile, Shape &meanShape, int SHAPE_SIZE){
    Shape *shapes;
    std::vector<std::string> imgList;
    int isize, step, count;
    char rootDir[128], fileName[128], ext[30], filePath[256];

    if(read_file_list(listFile, imgList) != 0){
        printf("READ FILE LIST ERROR\n");
        return 1;
    }

    isize = imgList.size();
    step = 1;
    if(isize > 10000) step = isize / 10000;

    shapes = new Shape[20000];

    printf("Load shape\n");
    count = 0;
    for(int i = 0; i < isize; i += step){
        const char *imgPath = imgList[i].c_str();

        analysis_file_path(imgPath, rootDir, fileName, ext);
        sprintf(filePath, "%s/%s.pts", rootDir, fileName);

        if(read_pts_file(filePath, shapes + count) != PTS_SIZE) {
            printf("Read pts error %s\n", filePath);
            return 2;
        }

        count++;
    }

    assert(count > 0);
    calculate_mean_shape_global(shapes, count, PTS_SIZE, SHAPE_SIZE, meanShape);

    {
        uint8_t *img = new uint8_t[SHAPE_SIZE * SHAPE_SIZE];
        memset(img, 255, sizeof(uint8_t) * SHAPE_SIZE * SHAPE_SIZE);
        write_shape(img, SHAPE_SIZE, SHAPE_SIZE, SHAPE_SIZE, meanShape, "log/meanShape.jpg");

        delete [] img;
    }

    delete [] shapes;

    return 0;
}


int read_positive_samples(const char *listFile, Shape meanShape, int WINW, int WINH, int mirrorFlag, SampleSet **resSet){
    std::vector<std::string> imgList;

    int isize, capacity;
    int step = 1;
    int winSize = HU_MAX(WINW, WINH);

    int count = 0;
    char rootDir[256], fileName[256], ext[30], filePath[256];

    SampleSet *set = new SampleSet;

    memset(set, 0, sizeof(SampleSet));

    if(read_file_list(listFile, imgList) != 0){
        printf("READ FILE LIST ERROR\n");
        return 1;
    }

    isize = imgList.size();

    capacity = isize;
    if(mirrorFlag) capacity *= 2;
    Sample **samples = new Sample*[capacity];

    memset(samples, 0, sizeof(Sample*) * capacity);

    printf("Load images\n");
    count = 0;
    for(int i = 0; i < isize; i++){
        const char *imgPath = imgList[i].c_str();
        cv::Mat img;
        cv::Rect rect;
        float scale, angle;

        Sample *sample = new Sample;

        img = cv::imread(imgPath, 0);
        if(img.empty()){
            printf("Can't open image %s\n", imgPath);
            return 1;
        }

        analysis_file_path(imgPath, rootDir, fileName, ext);
        sprintf(filePath, "%s/%s.pts", rootDir, fileName);

        memset(sample, 0, sizeof(Sample));

        if(read_pts_file(filePath, &sample->oriShape) != PTS_SIZE) {
            printf("Read pts error %s\n", filePath);
            return 2;
        }

        sample->curShape = sample->oriShape;
        extract_sample_from_image(img.data, img.cols, img.rows, img.step, meanShape, WINW, WINH, 0, sample);
        similarity_transform(meanShape, sample->curShape, sample->arg);
        strcpy(sample->patchName, fileName);

        samples[count++] = sample;

        if(mirrorFlag){
            Sample *samplev = new Sample;
            memset(samplev, 0, sizeof(Sample));
            samplev->iImg = new uint8_t[WINW * WINH];
            samplev->img = samplev->iImg;

            memcpy(samplev->img, sample->img, sizeof(uint8_t) * WINW * WINH);
            samplev->oriShape = sample->oriShape;

            mirror_sample(samplev->img, WINW, WINH, WINW, samplev->oriShape);
            sprintf(samplev->patchName, "%s_v", fileName);

            samples[count++] = samplev;
        }
        printf("%d\r", i), fflush(stdout);
    }

    set->samples = samples;
    set->capacity = capacity;
    set->ssize = count;

    set->meanShape = meanShape;

    set->ptsSize = set->meanShape.ptsSize;

    set->WINW = WINW;
    set->WINH = WINH;

    *resSet = set;

    return 0;
}


int read_negative_patch_samples(const char *listFile, SampleSet *negSet){
    std::vector<std::string> imgList;
    int size, ssize;
    int WINW, WINH;

    if(read_file_list(listFile, imgList) != 0)
        return 1;

    size = imgList.size();
    if(size == 0) return 2;

    reserve(negSet, negSet->ssize + size);

    WINW = negSet->WINW;
    WINH = negSet->WINH;

    for(int i = 0; i < size; i++){
        const char *imgPath = imgList[i].c_str();
        char rootDir[256], fileName[256], ext[30];

        cv::Mat img = cv::imread(imgPath, 0);
        assert(!img.empty());

        if(img.cols != WINW || img.rows != WINH)
            cv::resize(img, img, cv::Size(WINW, WINH));

        Sample *sample = new Sample;

        int border = BORDER_FACTOR * WINW;
        int stride = WINW + border * 2;

        sample->iImg = new uint8_t[(WINW + border * 2) * (WINH + border * 2)];
        sample->img = sample->iImg + border * stride + border;
        sample->stride = stride;

        for(int y = 0; y < WINH; y++)
            memcpy(sample->img + y * stride, img.data + y * img.step, sizeof(uint8_t) * WINW);

        analysis_file_path(imgPath, rootDir, fileName, ext);

        argument(negSet->meanShape, sample->curShape);
        similarity_transform(negSet->meanShape, sample->curShape, sample->arg);
        strcpy(sample->patchName, fileName);

        negSet->samples[negSet->ssize + i] = sample;
    }

    negSet->ssize += size;

    return 0;
}


void generate_transform_samples(SampleSet *oriSet, int times, SampleSet *resSet){

    int oriW = oriSet->WINW;
    int oriH = oriSet->WINH;
    int WINW = resSet->WINW;
    int WINH = resSet->WINH;

    int capacity = oriSet->ssize * times;

    cv::RNG rng(cv::getTickCount());

    uint8_t **buffers = new uint8_t*[times];

    buffers[0] = new uint8_t[times * oriW * oriH];

    for(int i = 1; i < times; i++)
        buffers[i] = buffers[0] + i * oriW * oriH;

    reserve(resSet, capacity);

    if(times == 1){
        for(int i = 0; i < oriSet->ssize; i++){
            Sample *oriSample = oriSet->samples[i];
            Sample *sample = new Sample;

            int border = BORDER_FACTOR * WINW;

            memset(sample, 0, sizeof(Sample));
            sample->oriShape = oriSample->oriShape;

            sample->curShape = sample->oriShape;
            extract_sample_from_image(oriSample->img, oriW, oriH, oriSample->stride, resSet->meanShape, WINW, WINH, border, sample);
            argument(resSet->meanShape, sample->curShape);

            //mean_filter_3x3(sample->iImg, WINW + border * 2, WINH + border * 2, sample->stride);

            similarity_transform(resSet->meanShape, sample->curShape, sample->arg);
            sprintf(sample->patchName, "%s", oriSample->patchName);

            resSet->samples[i] = sample;

            printf("%d\r", i), fflush(stdout);
        }
    }
    else {
        for(int i = 0; i < oriSet->ssize; i++){
            Sample *oriSample = oriSet->samples[i];

            for(int t = 0; t < times; t++){
                Sample *sample = new Sample;
                Shape oriShape = oriSample->oriShape;

                int border = BORDER_FACTOR * WINW;

                affine_sample(oriSample->img, oriW, oriH, oriW, oriShape, 1.0f, rng.uniform(-HU_PI / 9, HU_PI / 9), buffers[t]);
                transform_image(buffers[t], oriW, oriH, oriW);

                memset(sample, 0, sizeof(Sample));

                sample->oriShape = oriShape;
                sample->curShape = sample->oriShape;
                extract_sample_from_image(buffers[t], oriW, oriH, oriW, resSet->meanShape, WINW, WINH, border, sample);
                argument(resSet->meanShape, sample->curShape);

                //mean_filter_3x3(sample->iImg, WINW + border * 2, WINH + border * 2, sample->stride);

                similarity_transform(resSet->meanShape, sample->curShape, sample->arg);
                sprintf(sample->patchName, "%s_%d", oriSample->patchName, t);

                resSet->samples[i * times + t] = sample;
            }

            printf("%d\r", i), fflush(stdout);
        }
    }

    resSet->ssize = capacity;
    resSet->capacity = capacity;
    resSet->ptsSize = resSet->meanShape.ptsSize;

    delete [] buffers[0];
    delete [] buffers;
}


void calculate_residuals(SampleSet *set, double *rsdlsX, double *rsdlsY){
    int ptsSize = set->ptsSize;
    int WINW = set->WINW;
    int WINH = set->WINH;

    double sum = 0;

    assert(ptsSize == PTS_SIZE);

    for(int i = 0; i < set->ssize; i++){
        Sample *sample = set->samples[i];
        TranArgs arg;

        float dist = 0;

        similarity_transform(sample->curShape, set->meanShape, arg);

        for(int j = 0; j < ptsSize; j++){
            float dx = sample->oriShape.pts[j].x - sample->curShape.pts[j].x;
            float dy = sample->oriShape.pts[j].y - sample->curShape.pts[j].y;

            float sina = arg.ssina;
            float cosa = arg.scosa;

            float rsx = dx * cosa + dy * sina;
            float rsy = dy * cosa - dx * sina;

            int id = j * set->ssize + i;

            rsdlsX[id] = rsx;
            rsdlsY[id] = rsy;

            dist += sqrtf(rsx * rsx + rsy * rsy);
        }

        sum += dist / ptsSize;
    }

    printf("Residuals = %f\n", sum / set->ssize);
}


void calculate_residuals(SampleSet *set, int *idxs, int size, int pntIdx, float *rsdls){
    int ptsSize = set->ptsSize;
    int WINW = set->WINW;
    int WINH = set->WINH;

    int flag = (pntIdx % 2 == 0);

    pntIdx >>= 1;

    for(int i = 0; i < size; i++){
        Sample *sample = set->samples[idxs[i]];
        TranArgs arg;

        similarity_transform(sample->curShape, set->meanShape, arg);

        float dx = sample->oriShape.pts[pntIdx].x - sample->curShape.pts[pntIdx].x;
        float dy = sample->oriShape.pts[pntIdx].y - sample->curShape.pts[pntIdx].y;

        float sina = arg.ssina;
        float cosa = arg.scosa;

        if(flag){
            rsdls[i] = dx * cosa + dy * sina;
        }
        else {
            rsdls[i] = dy * cosa - dx * sina;
        }
    }
}


void refine_samples(SampleSet *set, float thresh, int flag){
    int isize = set->ssize;

    if(flag == 1){
        for(int i = 0; i < isize; i++){
            if(set->samples[i]->score <= thresh){
                HU_SWAP(set->samples[i], set->samples[isize - 1], Sample*);

                release(set->samples + isize - 1);
                set->samples[isize - 1] = NULL;

                i--;
                isize --;
            }
        }
    }
    else {
        for(int i = 0; i < isize; i++){
            Sample *sample = set->samples[i];

            sample->score = HU_MAX(sample->score, thresh);
        }
    }

    set->ssize = isize;
}


void reset_scores_and_args(SampleSet *set){
    for(int i = 0; i < set->ssize; i++){
        Sample *sample = set->samples[i];

        sample->score = 0;
        similarity_transform(set->meanShape, sample->curShape, sample->arg);
    }
}


void reserve(SampleSet *set, int capacity){
    if(set->capacity > capacity)
        return;

    Sample **samples = new Sample*[capacity];

    memset(samples, 0, sizeof(Sample*) * capacity);

    if(set->ssize > 0){
        memcpy(samples, set->samples, sizeof(Sample*) * set->ssize);
        delete [] set->samples;
    }

    set->samples = samples;
    set->capacity = capacity;
}


void add_sample_capacity_unchange(SampleSet *set, Sample *sample){
    if(set->capacity <= set->ssize){
        cv::RNG rng(cv::getTickCount());
        int len = 0.1 * set->ssize + 1;

        for(int i = 0; i < len; i++){
            int id = rng.uniform(0, set->ssize);

            HU_SWAP(set->samples[id], set->samples[set->ssize - 1], Sample*);
            release(&set->samples[set->ssize - 1]);

            set->ssize --;
        }
    }

    set->samples[set->ssize++] = sample;
}


void add_sample(SampleSet *set, Sample *sample){
    if(set->capacity > set->ssize){
        set->samples[set->ssize] = sample;
        set->ssize ++;
        return;
    }

    Sample **samples = new Sample*[set->ssize + 100];

    if(set->ssize > 0){
        memcpy(samples, set->samples, sizeof(Sample) * set->ssize);
        delete [] set->samples;
    }

    set->samples = samples;
    set->capacity += 100;

    set->samples[set->ssize ++] = sample;
}


void add_samples(SampleSet *dset, SampleSet *sset){
    if(sset->ssize == 0)
        return ;

    assert(dset->WINW == sset->WINW);
    assert(dset->WINH == sset->WINH);

    int capacity = dset->ssize + sset->ssize;

    if(capacity > dset->capacity){
        Sample **samples = new Sample*[capacity];

        if(dset->ssize > 0){
            memcpy(samples, dset->samples, sizeof(Sample*) * dset->ssize);
            delete [] dset->samples;
        }

        dset->samples = samples;
        dset->capacity = capacity;
    }


    for(int i = 0; i < sset->ssize; i++)
        dset->samples[dset->ssize++] = sset->samples[i];

    delete [] sset->samples;
    sset->samples = NULL;

    sset->ssize = 0;
    sset->capacity = 0;
}


void random_samples(SampleSet *set){
    static cv::RNG rng(cv::getTickCount());

    for(int i = 0; i < set->ssize; i++){
        int id1 = rng.uniform(0, set->ssize);
        int id2 = rng.uniform(0, set->ssize);

        if(id1 == id2) continue;

        HU_SWAP(set->samples[id1], set->samples[id2], Sample*);
    }
}


int save(const char *filePath, SampleSet *set)
{
    FILE *fout = fopen(filePath, "wb");

    if(fout == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    int ret;

    ret = fwrite(&set->ssize, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&set->ptsSize, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&set->WINW, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&set->WINH, sizeof(int), 1, fout); assert(ret == 1);

    ret = fwrite(&set->meanShape, sizeof(Shape), 1, fout); assert(ret == 1);

    int sq = set->WINW * set->WINH;

    for(int i = 0; i < set->ssize; i++){
        Sample *sample = set->samples[i];

        ret = fwrite(sample->img, sizeof(uint8_t), sq, fout);
        assert(ret == sq);

        ret = fwrite(&sample->oriShape, sizeof(Shape), 1, fout);
        assert(ret == 1);

        ret = fwrite(sample->patchName, sizeof(char), 100, fout);
        assert(ret == 100);
    }

    fclose(fout);

    return 0;
}


int load(const char *filePath, SampleSet **resSet){
    FILE *fin = fopen(filePath, "rb");
    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    SampleSet *set = new SampleSet;
    memset(set, 0, sizeof(SampleSet));

    int ret;

    ret = fread(&set->ssize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&set->ptsSize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&set->WINW, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&set->WINH, sizeof(int), 1, fin); assert(ret == 1);

    printf("%d %d %d %d\n", set->ssize, set->ptsSize, set->WINW, set->WINH);

    set->samples = new Sample*[set->ssize];

    memset(set->samples, 0, sizeof(Sample*) * set->ssize);

    ret = fread(&set->meanShape, sizeof(Shape), 1, fin); assert(ret == 1);

    int sq = set->WINW * set->WINH;

    for(int i = 0; i < set->ssize; i++){

        Sample *sample = new Sample;
        float angle, scale;

        memset(sample, 0, sizeof(Sample));

        sample->iImg = new uint8_t[sq];
        sample->img = sample->iImg;
        sample->stride = set->WINW;

        ret = fread(sample->img, sizeof(uint8_t), sq, fin);
        assert(ret == sq);

        ret = fread(&sample->oriShape, sizeof(Shape), 1, fin);
        assert(ret == 1);

        ret = fread(sample->patchName, sizeof(char), 100, fin);
        assert(ret == 100);

        argument(set->meanShape, sample->curShape);
        similarity_transform(set->meanShape, sample->curShape, sample->arg);

        set->samples[i] = sample;
    }

    set->capacity = set->ssize;

    *resSet = set;

    fclose(fin);

    return 0;
}


void write_images(SampleSet *set, const char *outDir, int step){
    char command[512];

    sprintf(command, "mkdir -p %s", outDir);
    int ret = system(command);

    int WINW = set->WINW;
    int WINH = set->WINH;

    for(int i = 0; i < set->ssize; i += step){
        Sample *sample = set->samples[i];
        Shape shape = sample->curShape;
        char filePath[256];

        cv::Mat sImg(WINH, WINW, CV_8UC1, sample->img, sample->stride);

        //*
        cv::cvtColor(sImg, sImg, cv::COLOR_GRAY2BGR);

        for(int p = 0; p < shape.ptsSize; p++)
            cv::circle(sImg, cv::Point2f(shape.pts[p].x, shape.pts[p].y), 1, cv::Scalar(0, 255, 0), -1);
        //*/
        sprintf(filePath, "%s/%s.bmp", outDir, sample->patchName);
        cv::imwrite(filePath, sImg);
    }
}


void release_data(Sample *sample){
    if(sample->img != NULL)
        delete [] sample->iImg;

    sample->img = NULL;
    sample->iImg = NULL;
}


void release(Sample **sample){
    if(*sample != NULL){
        release_data(*sample);
        delete *sample;
    }

    *sample = NULL;
}


void release_data(SampleSet *set){
    if(set->samples != NULL){
        for(int i = 0; i < set->ssize; i++){
            if(set->samples[i] != NULL)
                release(&set->samples[i]);

            set->samples[i] = NULL;
        }

        delete [] set->samples;
    }
    set->samples = NULL;
    set->ssize = 0;
    set->capacity = 0;
}


void release(SampleSet **set){
    if(*set != NULL){
        release_data(*set);
        delete *set;
    }

    *set = NULL;
}


void print_info(SampleSet *set){
    printf("ssize: %d, capacity: %d\n", set->ssize, set->capacity);
    printf("ptsSize: %d, WINW: %d, WINH: %d\n", set->ptsSize, set->WINW, set->WINH);
}


static void vertical_mirror(uint8_t *img, int width, int height, int stride){
    int cx = width / 2;

    for(int y = 0; y < height; y++){
        for(int x = 0; x < cx; x++)
            HU_SWAP(img[x], img[width - 1 - x], uint8_t);

        img += stride;
    }
}


static void vertical_mirror(Shape &shape, int width){
    int ptsSize = shape.ptsSize;

    for(int i = 0; i < ptsSize; i++)
        shape.pts[i].x = width - 1 - shape.pts[i].x;

    if(ptsSize == 68){
        int idxs1[29] = {  0,  1,  2,  3,  4,  5,  6,  7,
                        17, 18, 19, 20, 21,
                        31, 32,
                        36, 37, 38, 39, 40, 41,
                        48, 49, 50, 58, 59, 60, 61, 67 };

        int idxs2[29] = { 16, 15, 14, 13, 12, 11, 10,  9,
                        26, 25, 24, 23, 22,
                        35, 34,
                        45, 44, 43, 42, 47, 46,
                        54, 53, 52, 56, 55, 64, 63, 65 };


        for(int i = 0; i < 29; i++){
            int id1 = idxs1[i];
            int id2 = idxs2[i];

            HU_SWAP(shape.pts[id1], shape.pts[id2], HPoint2f);
        }

    }
    else if(ptsSize == 51){
        int idxs1[21] = {
            0, 1, 2, 3, 4,
            14, 15,
            19, 20, 21, 22, 23, 24,
            31, 32, 33, 41, 42, 43, 44, 50};

        int idxs2[21] = {
                        9, 8, 7, 6, 5,
                        18, 17,
                        28, 27, 26, 25, 30, 29,
                        37, 36, 35, 39, 38, 47, 46, 48};

        for(int i = 0; i < 21; i++){
            int id1 = idxs1[i];
            int id2 = idxs2[i];

            HU_SWAP(shape.pts[id1], shape.pts[id2], HPoint2f);
        }
    }
    else if(ptsSize == 101){
        int idxs1[46] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 95,
            63, 64, 65, 66, 67, 68,
            75, 76, 77, 85, 86, 87, 88, 94,
        };
        int idxs2[46] = {
            18, 17, 16, 15, 14, 13, 12, 11, 10,
            34, 33, 32, 31, 30, 29, 38, 37, 36, 35,
            57, 56, 55, 54, 53, 52, 51, 62, 61, 60, 59, 58, 96,
            74, 73, 72, 71, 70, 69,
            81, 80, 79, 83, 82, 91, 90, 92
        };

        for(int i = 0; i < 46; i++){
            int id1 = idxs1[i];
            int id2 = idxs2[i];

            HU_SWAP(shape.pts[id1], shape.pts[id2], HPoint2f);
        }
    }
    else{
        printf("NO PTS FORMAT: %d\n", ptsSize);
        exit(-1);
    }
}


void mirror_sample(uint8_t *img, int width, int height, int stride, Shape &shape){
    vertical_mirror(img, width, height, stride);
    vertical_mirror(shape, width);
}


void show_shape(Shape &shape, const char *winName){
    int ptsSize = shape.ptsSize;

    float minx = FLT_MAX, maxx = -FLT_MAX;
    float miny = FLT_MAX, maxy = -FLT_MAX;

    for(int i = 0; i < ptsSize; i++){
        minx = HU_MIN(minx, shape.pts[i].x);
        maxx = HU_MAX(maxx, shape.pts[i].x);

        miny = HU_MIN(miny, shape.pts[i].y);
        maxy = HU_MAX(maxy, shape.pts[i].y);
    }

    int faceSize = HU_MAX(maxx - minx + 1, maxy - miny + 1) + 20;

    faceSize = HU_MAX(96, faceSize);
    printf("%d\n", faceSize);

    cv::Mat img(faceSize, faceSize, CV_8UC3, cv::Scalar::all(255));
    for(int i = 0; i < ptsSize; i++)
        cv::circle(img, cv::Point2f(shape.pts[i].x - minx + 10, shape.pts[i].y - miny + 10), 2, cv::Scalar(0, 0, 255), -1);

    cv::imshow(winName, img);
    cv::waitKey();
}


void show_shape(cv::Mat &img, Shape &shape, const char *winName){
    cv::Mat cImg;
    int ptsSize;

    if(img.channels() == 3)
        img.copyTo(cImg);
    else
        cv::cvtColor(img, cImg, cv::COLOR_GRAY2BGR);

    ptsSize = shape.ptsSize;

    for(int i = 0; i < ptsSize; i++){
        cv::circle(cImg, cv::Point2f(shape.pts[i].x, shape.pts[i].y), 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::imshow(winName, cImg);
    cv::waitKey();
}


void show_shape(uint8_t *img, int width, int height, int stride, Shape &shape, const char *winName){
    cv::Mat src, cImg;
    int ptsSize = shape.ptsSize;

    src = cv::Mat(height, width, CV_8UC1, img, stride);

    cv::cvtColor(src, cImg, cv::COLOR_GRAY2BGR);

    for(int i = 0; i < ptsSize; i++)
        cv::circle(cImg, cv::Point2f(shape.pts[i].x, shape.pts[i].y), 3, cv::Scalar(0, 255, 0), -1);

    cv::imshow(winName, cImg);
    cv::waitKey();
}


void write_shape(uint8_t *img, int width, int height, int stride, Shape &shape, const char *fileName){
    cv::Mat src, cImg;
    int ptsSize = shape.ptsSize;

    src = cv::Mat(height, width, CV_8UC1, img, stride);

    cv::cvtColor(src, cImg, cv::COLOR_GRAY2BGR);

    for(int i = 0; i < ptsSize; i++)
        cv::circle(cImg, cv::Point2f(shape.pts[i].x, shape.pts[i].y), 1, cv::Scalar(0, 255, 0), -1);

    cv::imwrite(fileName, cImg);
}


void affine_shape(Shape *shapeSrc, HPoint2f cen1, Shape *shapeRes, HPoint2f cen2, float scale, float angle){
    float sina = sin(angle) * scale;
    float cosa = cos(angle) * scale;

    for(int i = 0; i < shapeSrc->ptsSize; i++){
        float x = shapeSrc->pts[i].x - cen1.x;
        float y = shapeSrc->pts[i].y - cen1.y;

        shapeRes->pts[i].x =  x * cosa + y * sina + cen2.x;
        shapeRes->pts[i].y = -x * sina + y * cosa + cen2.y;
    }

    shapeRes->ptsSize = shapeSrc->ptsSize;
}


#define FIX_INTER_POINT 14

void affine_sample(uint8_t *img, int width, int height, int stride, Shape &shape, float scale, float angle, uint8_t *dst){
    int FIX_ONE = 1 << FIX_INTER_POINT;
    int FIX_0_5 = FIX_ONE >> 1;

    float sina = sinf(-angle) / scale;
    float cosa = cosf(-angle) / scale;

    int id = 0;

    int dstw = width;
    int dsth = height;
    int dsts = stride;

    int *xtable = new int[(dstw << 1) + (dsth << 1)]; assert(xtable != NULL);
    int *ytable = xtable + (dstw << 1);

    HPoint2f center;
    center.x = width >> 1;
    center.y = height >> 1;

    int fcx = center.x * FIX_ONE;
    int fcy = center.y * FIX_ONE;

    for(int i = 0; i < dsth; i++){
        int idx = i << 1;

        float y = (i - center.y);

        ytable[idx]     = y * sina * FIX_ONE + fcx;
        ytable[idx + 1] = y * cosa * FIX_ONE + fcy;
    }

    for(int i = 0; i < dstw; i++){
        int idx = i << 1;

        float x = (i - center.x);

        xtable[idx]     = x * sina * FIX_ONE;
        xtable[idx + 1] = x * cosa * FIX_ONE;
    }

    memset(dst, 0, sizeof(uint8_t) * width * height);

    id = 0;
    for(int y = 0; y < dsth; y++){
        int idx = y << 1;

        int ys = ytable[idx]    ;
        int yc = ytable[idx + 1];

        for(int x = 0; x < dstw; x++){
            idx = x << 1;

            int xs = xtable[idx];
            int xc = xtable[idx + 1];

            int fx =  xc + ys;
            int fy = -xs + yc;

            int x0 = fx >> FIX_INTER_POINT;
            int y0 = fy >> FIX_INTER_POINT;

            int wx = fx - (x0 << FIX_INTER_POINT);
            int wy = fy - (y0 << FIX_INTER_POINT);

            if(x0 < 0 || x0 >= width || y0 < 0 || y0 >= height)
                continue;

            assert(wx <= FIX_ONE && wy <= FIX_ONE);

            uint8_t *ptr1 = img + y0 * stride + x0;
            uint8_t *ptr2 = ptr1 + stride;

            uint8_t value0 = ((ptr1[0] << FIX_INTER_POINT) + (ptr1[1] - ptr1[0]) * wx + FIX_0_5) >> FIX_INTER_POINT;
            uint8_t value1 = ((ptr2[0] << FIX_INTER_POINT) + (ptr2[1] - ptr2[0]) * wx + FIX_0_5) >> FIX_INTER_POINT;

            dst[id + x] = ((value0 << FIX_INTER_POINT) + (value1 - value0) * wy + FIX_0_5) >> FIX_INTER_POINT;
        }

        id += dsts;
    }

    delete [] xtable;

    affine_shape(&shape, center, &shape, center, scale, angle);
}


float shape_euler_distance(Shape &shapeA, Shape &shapeB){
    float dist = 0;

    for(int i = 0; i < shapeA.ptsSize; i++){
        HPoint2f ptsa = shapeA.pts[i];
        HPoint2f ptsb = shapeB.pts[i];

        dist += sqrtf(powf(ptsa.x - ptsb.x, 2) + powf(ptsa.y - ptsb.y, 2));
    }

    return dist;
}


float shape_euler_distance(Shape &shapeA, Shape &shapeB, Shape &meanShape){
    float dist = 0;
    int ptsSize = meanShape.ptsSize;

    TranArgs arg;

    similarity_transform(shapeA, meanShape, arg);

    for(int i = 0; i < ptsSize; i++){
        float dx = shapeA.pts[i].x - shapeB.pts[i].x;
        float dy = shapeA.pts[i].y - shapeB.pts[i].y;

        float rx = dx * arg.scosa + dy * arg.ssina;
        float ry = dy * arg.scosa - dx * arg.ssina;

        dist += sqrtf(rx * rx + ry * ry);
    }

    return dist;
}


#define MAX_SCORE 0.5f
void initial_positive_set_scores_by_distance(SampleSet *posSet){
    int ssize = posSet->ssize;
    float *dists = new float[ssize];

    float minv = FLT_MAX, maxv = -FLT_MAX;
    float factor;

    for(int i = 0; i < ssize; i++){
        Sample *sample = posSet->samples[i];

        float dist = shape_euler_distance(sample->curShape, sample->oriShape, posSet->meanShape);

        minv = HU_MIN(minv, dist);
        maxv = HU_MAX(maxv, dist);

        dists[i] = dist;
    }

    factor = MAX_SCORE / (maxv - minv);

    for(int i = 0; i < ssize; i++){
        Sample *sample = posSet->samples[i];

        sample->score = MAX_SCORE - (dists[i] - minv) * factor;
    }

    delete [] dists;
}

