#include "sample.h"


void mirror_sample(uint8_t *img, int width, int height, int stride, Shape &shape);

int read_pts_file(const char *filePath, float *shape, int &ptsSize)
{
    FILE *fin = fopen(filePath, "r");

    char line[256];
    char *ret;

    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 0;
    }

    ret = fgets(line, 255, fin);
    ret = fgets(line, 255, fin);

    sscanf(line, "n_points:  %d", &ptsSize);

    assert(ptsSize <= MAX_PTS_SIZE);

    ret = fgets(line, 255, fin);

    for(int i = 0; i < ptsSize; i++){
        if(fgets(line, 255, fin) == NULL) {
            fclose(fin);
            printf("END of FILE: %s\n", filePath);
            return 0;
        }

        int ret = sscanf(line, "%f %f\n", shape + i, shape + i + ptsSize);

        if(ret == 0) break;
    }

    fclose(fin);


    return ptsSize;
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
        fprintf(fout, "%f %f\n", shape[i], shape[i + ptsSize]);

    fprintf(fout, "}");
    fclose(fout);

    return 0;
}


void calculate_mean_shape_global(Shape *shapes, int size, int ptsSize, int SHAPE_SIZE, Shape &meanShape)
{
    double cxs[MAX_PTS_SIZE], cys[MAX_PTS_SIZE];

    float minx, miny, maxx, maxy;
    float cx, cy, w, h, faceSize, scale;

    memset(cxs, 0, sizeof(double) * MAX_PTS_SIZE);
    memset(cys, 0, sizeof(double) * MAX_PTS_SIZE);

    for(int i = 0; i < size; i++){
        Shape &shape = shapes[i];

        minx = FLT_MAX; maxx = -FLT_MAX;
        miny = FLT_MAX; maxy = -FLT_MAX;

        for(int j = 0; j < ptsSize; j++){
            float x = shape.pts[j];
            float y = shape.pts[j + ptsSize];

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
            cxs[j] += (shape.pts[j] - cx) * scale;
            cys[j] += (shape.pts[j + ptsSize] - cy) * scale;
        }
    }

    minx =  FLT_MAX, miny =  FLT_MAX;
    maxx = -FLT_MAX, maxy = -FLT_MAX;

    meanShape.ptsSize = ptsSize;

    for(int j = 0; j < ptsSize; j++){
        float x = cxs[j] / size;
        float y = cys[j] / size;

        minx = HU_MIN(minx, x);
        maxx = HU_MAX(maxx, x);

        miny = HU_MIN(miny, y);
        maxy = HU_MAX(maxy, y);

        meanShape.pts[j] = x;
        meanShape.pts[j + ptsSize] = y;
    }

    w = maxx - minx + 1;
    h = maxy - miny + 1;

    cx = (maxx + minx) * 0.5f;
    cy = (maxy + miny) * 0.5f;

    faceSize = HU_MAX(w, h);

    scale = SHAPE_SIZE / faceSize;

    for(int j = 0; j < ptsSize; j++){
        meanShape.pts[j] = (meanShape.pts[j] - cx) * scale;
        meanShape.pts[j + ptsSize] = (meanShape.pts[j + ptsSize] - cy) * scale;
    }
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
    if(isize < 3000){
        printf("Samples  not enough, must more than 3000\n");
        return  2;
    }

    step = 1;
    if(isize > 10000) step = isize / 10000;

    shapes = new Shape[20000];

    printf("Load shape\n");
    count = 0;
    for(int i = 0; i < isize; i += step){
        const char *imgPath = imgList[i].c_str();

        analysis_file_path(imgPath, rootDir, fileName, ext);
        sprintf(filePath, "%s/%s.pts", rootDir, fileName);

        if(read_pts_file(filePath, shapes[count].pts, shapes[count].ptsSize) == 0) {
            printf("Read pts error %s\n", filePath);
            return 3;
        }

        count++;
    }

    assert(count > 0);
    calculate_mean_shape_global(shapes, count, shapes[0].ptsSize, SHAPE_SIZE, meanShape);

    delete [] shapes;

    int ret = system("mkdir -p log");

    write_shape(meanShape, "log/meanShape.jpg");

    return 0;
}


void similarity_transform(Shape &src, Shape &dst, TranArgs &arg){
    int ptsSize = src.ptsSize;
    float cx1 = 0.0f, cy1 = 0.0f;
    float cx2 = 0.0f, cy2 = 0.0f;

    int i, j;

    for(i = 0, j = ptsSize; i <= ptsSize - 4; i += 4, j += 4){
        cx1 += src.pts[i];
        cx1 += src.pts[i + 1];
        cx1 += src.pts[i + 2];
        cx1 += src.pts[i + 3];

        cy1 += src.pts[j];
        cy1 += src.pts[j + 1];
        cy1 += src.pts[j + 2];
        cy1 += src.pts[j + 3];
    }

    for(; i < ptsSize; i++, j++){
        cx1 += src.pts[i];
        cy1 += src.pts[j];
    }

    cx1 /= ptsSize;
    cy1 /= ptsSize;

    for(i = 0, j = ptsSize; i <= ptsSize - 4; i += 4, j += 4){
        cx2 += dst.pts[i];
        cx2 += dst.pts[i + 1];
        cx2 += dst.pts[i + 2];
        cx2 += dst.pts[i + 3];

        cy2 += dst.pts[j];
        cy2 += dst.pts[j + 1];
        cy2 += dst.pts[j + 2];
        cy2 += dst.pts[j + 3];
    }

    for(; i < ptsSize; i++, j++){
        cx2 += dst.pts[i];
        cy2 += dst.pts[j];
    }

    cx2 /= ptsSize;
    cy2 /= ptsSize;

    float ssina = 0.0f, scosa = 0.0f, num = 0.0f;
    float var1 = 0.0f, var2 = 0.0f;

    for(int i = 0, j = ptsSize; i < ptsSize; i++, j++){
        float sx = src.pts[i] - cx1;
        float sy = src.pts[j] - cy1;

        float dx = dst.pts[i] - cx2;
        float dy = dst.pts[j] - cy2;

        var1 += sqrtf(sx * sx + sy * sy);
        var2 += sqrtf(dx * dx + dy * dy);

        ssina += (sy * dx - sx * dy);
        scosa += (sx * dx + sy * dy);
    }

    num = 1.0f / sqrtf(ssina * ssina + scosa * scosa);

    ssina *= num;
    scosa *= num;

    arg.angle = asinf(ssina);
    arg.scale = var2 / var1;

    arg.cen1.x = cx1;
    arg.cen1.y = cy1;
    arg.cen2.x = cx2;
    arg.cen2.y = cy2;
}


void extract_sample_from_image(uint8_t *img, int width, int height, int stride, Shape &meanShape, int WINW, int WINH, Sample* sample){
    Shape shape = sample->oriShape;
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

    assert(shape.ptsSize == meanShape.ptsSize);

    similarity_transform(shape, meanShape, arg);

    scale = arg.scale;

    swidth  = width * scale + 0.5f;
    sheight = height * scale + 0.5f;
    sstride = swidth;
    sImg = new uint8_t[sstride * sheight];

    resizer_bilinear_gray(img, width, height, stride, sImg, swidth, sheight, sstride);

    for(int i = 0, j = shape.ptsSize; i < shape.ptsSize; i++, j++){
        shape.pts[i] *= scale;

        minx = HU_MIN(minx, shape.pts[i]);
        maxx = HU_MAX(maxx, shape.pts[i]);

        shape.pts[j] *= scale;
        miny = HU_MIN(miny, shape.pts[j]);
        maxy = HU_MAX(maxy, shape.pts[j]);
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

    if(sample->img == NULL)
        sample->img = new uint8_t[WINW * WINH];

    memset(sample->img, 0, sizeof(uint8_t) * WINW * WINH);

    w = WINW - bl - br;
    h = WINH - bt - bb;

    uint8_t *ptrDst = sample->img + bt * WINW + bl;
    uint8_t *ptrSrc = sImg + y0 * sstride + x0;

    for(int y = 0; y < h; y++)
        memcpy(ptrDst + y * WINW, ptrSrc + y * sstride, sizeof(uint8_t) * w);

    for(int i = 0, j = shape.ptsSize; i < shape.ptsSize; i++, j++){
        shape.pts[i] += (bl - x0);
        shape.pts[j] += (bt - y0);
    }

    sample->oriShape = shape;

    delete [] sImg;
}


int read_samples(const char *listFile, Shape &meanShape, int WINW, int WINH, int mirrorFlag, SampleSet **resSet){
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

        memset(sample, 0, sizeof(Sample));

        img = cv::imread(imgPath, 0);
        if(img.empty()){
            printf("Can't open image %s\n", imgPath);
            return 1;
        }

        analysis_file_path(imgPath, rootDir, fileName, ext);
        sprintf(filePath, "%s/%s.pts", rootDir, fileName);

        if(read_pts_file(filePath, sample->oriShape.pts, sample->oriShape.ptsSize) == 0) {
            printf("Read pts error %s\n", filePath);
            return 2;
        }

        extract_sample_from_image(img.data, img.cols, img.rows, img.step, meanShape, WINW, WINH, sample);

        strcpy(sample->patchName, fileName);

        samples[count++] = sample;

        if(mirrorFlag){
            Sample *samplev = new Sample;
            memset(samplev, 0, sizeof(Sample));
            samplev->img = new uint8_t[WINW * WINH];

            memcpy(samplev->img, sample->img, sizeof(uint8_t) * WINW * WINH);
            samplev->oriShape = sample->oriShape;

            mirror_sample(samplev->img, WINW, WINH, WINW, samplev->oriShape);

            sprintf(samplev->patchName, "%s_v", fileName);

            samples[count++] = samplev;
        }

        printf("%d\r", i), fflush(stdout);
    }

    set->samples = samples;
    set->ssize = count;

    set->meanShape = meanShape;

    set->ptsSize = set->meanShape.ptsSize;

    set->WINW = WINW;
    set->WINH = WINH;

    *resSet = set;

    write_images(set, set->ssize / 3000, "log/sample");

    assert(save(SAMPLE_FILE, set) == 0);

    return 0;
}


int save(const char *filePath, SampleSet *set){
    FILE *fout = fopen(filePath, "wb");
    if(fout == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    if(set == NULL || set->ssize == 0){
        printf("Sample set empty\n");
        return 2;
    }

    int ret;

    int WINW = set->WINW;
    int WINH = set->WINH;
    int sq = WINW * WINH;

    ret = fwrite(&set->ssize, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&set->WINW, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&set->WINH, sizeof(int), 1, fout); assert(ret == 1);
    ret = fwrite(&set->meanShape, sizeof(Shape), 1, fout); assert(ret == 1);

    for(int i = 0; i < set->ssize; i++){
        Sample *sample = set->samples[i];

        ret = fwrite(&sample->oriShape, sizeof(Shape), 1, fout); assert(ret == 1);
        ret = fwrite(sample->img, sizeof(uint8_t), sq, fout); assert(ret == sq);
        ret = fwrite(sample->patchName, sizeof(char), 100, fout); assert(ret == 100);
    }

    fclose(fout);

    return 0;
}


int load(const char *filePath, SampleSet **resSet){
    FILE *fin = NULL;
    SampleSet *set = NULL;
    int ret, sq;

    fin = fopen(filePath, "rb");

    if(fin == NULL){
        printf("Can't open file %s\n", filePath);
        return 1;
    }

    set = new SampleSet;

    memset(set, 0, sizeof(SampleSet));

    ret = fread(&set->ssize, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&set->WINW, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&set->WINH, sizeof(int), 1, fin); assert(ret == 1);
    ret = fread(&set->meanShape, sizeof(Shape), 1, fin); assert(ret == 1);

    set->ptsSize = set->meanShape.ptsSize;
    set->samples = new Sample*[set->ssize];

    sq = set->WINW * set->WINH;

    for(int i = 0; i < set->ssize; i++){
        Sample *sample = new Sample;

        set->samples[i] = sample;

        sample->img = new uint8_t[sq];

        ret = fread(&sample->oriShape, sizeof(Shape), 1, fin); assert(ret == 1);
        ret = fread(sample->img, sizeof(uint8_t), sq, fin); assert(ret == sq);
        ret = fread(sample->patchName, sizeof(char), 100, fin); assert(ret == 100);
    }

    fclose(fin);

    *resSet = set;

    write_images(set, set->ssize / 4000, "log/sample");
    write_shape(set->meanShape, "log/meanshape.jpg");

    return 0;
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
        shape.pts[i] = width - 1 - shape.pts[i];

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

            HU_SWAP(shape.pts[id1], shape.pts[id2], float);
            HU_SWAP(shape.pts[id1 + ptsSize], shape.pts[id2 + ptsSize], float);
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

            HU_SWAP(shape.pts[id1], shape.pts[id2], float);
            HU_SWAP(shape.pts[id1 + ptsSize], shape.pts[id2 + ptsSize], float);
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


void affine_shape(Shape &shapeSrc, Shape &shapeDst, TranArgs &arg){
    int ptsSize = shapeSrc.ptsSize;

    float sina = sin(arg.angle) * arg.scale;
    float cosa = cos(arg.angle) * arg.scale;

    float *ptrSx = shapeSrc.pts;
    float *ptrSy = shapeSrc.pts + ptsSize;

    float *ptrDx = shapeDst.pts;
    float *ptrDy = shapeDst.pts + ptsSize;

    for(int i = 0; i < ptsSize; i++){
        float x = ptrSx[i] - arg.cen1.x;
        float y = ptrSy[i] - arg.cen1.y;

        ptrDx[i] =  x * cosa + y * sina + arg.cen2.x;
        ptrDy[i] = -x * sina + y * cosa + arg.cen2.y;
    }

    shapeDst.ptsSize = ptsSize;
}

#define FIX_INTER_POINT 14

void affine_sample(uint8_t *img, int width, int height, int stride, Shape &shape, TranArgs &arg)
{
    uint8_t *imgBuffer = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    memset(imgBuffer, 0, sizeof(uint8_t) * width * height);

    int FIX_ONE = 1 << FIX_INTER_POINT;
    int FIX_0_5 = FIX_ONE >> 1;

    int dstw = width;
    int dsth = height;
    int dsts = width;

    uint8_t *dst = imgBuffer;

    float sina = sinf(-arg.angle) / arg.scale;
    float cosa = cosf(-arg.angle) / arg.scale;

    int id = 0;

    int *xtable = new int[(dstw << 1) + (dsth << 1)]; assert(xtable != NULL);
    int *ytable = xtable + (dstw << 1);


    int fcx = arg.cen1.x * FIX_ONE;
    int fcy = arg.cen1.y * FIX_ONE;

    for(int i = 0; i < dsth; i++){
        int idx = i << 1;

        float y = (i - arg.cen2.y);

        ytable[idx]     = y * sina * FIX_ONE + fcx;
        ytable[idx + 1] = y * cosa * FIX_ONE + fcy;
    }

    for(int i = 0; i < dstw; i++){
        int idx = i << 1;

        float x = (i - arg.cen2.x);

        xtable[idx]     = x * sina * FIX_ONE;
        xtable[idx + 1] = x * cosa * FIX_ONE;
    }

    affine_shape(shape, shape, arg);


    id = 0;
    for(int y = 0; y < dsth; y++){
        int idy = y << 1;

        int ys = ytable[idy]    ;
        int yc = ytable[idy + 1];

        for(int x = 0; x < dstw; x++){
            int idx = x << 1;

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


            uint8_t *ptr1 = img + y0 * stride + x0;
            uint8_t *ptr2 = ptr1 + stride;

            uint8_t value0 = ((ptr1[0] << FIX_INTER_POINT) + (ptr1[1] - ptr1[0]) * wx + FIX_0_5) >> FIX_INTER_POINT;
            uint8_t value1 = ((ptr2[0] << FIX_INTER_POINT) + (ptr2[1] - ptr2[0]) * wx + FIX_0_5) >> FIX_INTER_POINT;

            dst[id + x] = ((value0 << FIX_INTER_POINT) + (value1 - value0) * wy + FIX_0_5) >> FIX_INTER_POINT;
        }

        id += dsts;
    }

    delete [] xtable;

    id = 0;

    for(int y = 0; y < height; y++, id += width, img += stride)
        memcpy(img, imgBuffer + id, sizeof(uint8_t) * width);

    delete [] imgBuffer;

}


HRect get_shape_rect(Shape &shape){
    float minx = shape.pts[0], maxx = minx;
    float miny = shape.pts[shape.ptsSize], maxy = miny;

    HRect rect;

    for(int i = 1; i < shape.ptsSize; i++){
        float x = shape.pts[i];
        float y = shape.pts[i + shape.ptsSize];

        minx = HU_MIN(minx, x);
        maxx = HU_MAX(maxx, x);
        miny = HU_MIN(miny, y);
        maxy = HU_MAX(maxy, y);
    }

    float cx = 0.5f * (minx + maxx);
    float cy = 0.5f * (miny + maxy);

    float faceSize = HU_MAX(maxx - minx + 1, maxy - miny + 1);

    rect.x = cx - faceSize * 0.5f;
    rect.y = cy - faceSize * 0.5f;
    rect.width = rect.height = faceSize + 0.5f;

    return rect;
}


void release_data(Sample *sample){
    if(sample == NULL )
        return;

    if(sample->img != NULL)
        delete [] sample->img;
    sample->img = NULL;
}


void release(Sample **sample){
    if(*sample == NULL)
        return;

    release_data(*sample);
    delete *sample;
    *sample = NULL;
}


void release_data(SampleSet *set){
    if(set == NULL) return;

    if(set->samples != NULL){
        for(int i = 0; i < set->ssize; i++){
            release(&set->samples[i]);
        }

        delete [] set->samples;
    }

    set->samples = NULL;
    set->ssize = 0;
}


void release(SampleSet **set){
    if(*set == NULL)
        return ;

    release_data(*set);

    delete *set;
    *set = NULL;
}


void show_rect(uint8_t *img, int width, int height, int stride, HRect &rect, const char *winName){
    cv::Mat sImg(height, width, CV_8UC1, img, stride);
    cv::Mat colorImg;

    cv::cvtColor(sImg, colorImg, cv::COLOR_GRAY2BGR);

    cv::rectangle(colorImg, cv::Rect(rect.x, rect.y, rect.width, rect.height), cv::Scalar(0, 255, 0), 2);
    cv::imshow(winName, colorImg);
    cv::waitKey();
}


void show_shape(Shape &shape, const char *winName){
    HRect rect = get_shape_rect(shape);

    cv::Mat img(rect.height + 10, rect.width + 10, CV_8UC3, cv::Scalar(255, 255, 255));

    int ptsSize = shape.ptsSize;

    for(int p = 0; p < ptsSize; p++){
        float x = shape.pts[p] - rect.x + 5;
        float y = shape.pts[p + ptsSize] - rect.y + 5;

        cv::circle(img, cv::Point2f(x, y), 2, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow(winName, img);
    cv::waitKey();
}



void show_shape(uint8_t *img, int width, int height, int stride, Shape &shape, const char *winName){
    cv::Mat sImg(height, width, CV_8UC1, img, stride);
    cv::Mat colorImg;

    cv::cvtColor(sImg, colorImg, cv::COLOR_GRAY2BGR);

    int ptsSize = shape.ptsSize;

    for(int p = 0; p < ptsSize; p++){
        cv::circle(colorImg, cv::Point2f(shape.pts[p], shape.pts[p + ptsSize]), 2, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow(winName, colorImg);
    cv::waitKey();
}


void write_shape(Shape &shape, const char *filePath){
    HRect rect = get_shape_rect(shape);

    int cx = rect.x + (rect.width >> 1);
    int cy = rect.y + (rect.height >> 1);

    cv::Mat img(rect.height + 10, rect.width + 10, CV_8UC3, cv::Scalar(255, 255, 255));

    int ptsSize = shape.ptsSize;

    for(int p = 0; p < ptsSize; p++){
        float x = shape.pts[p] - cx + 5;
        float y = shape.pts[p + ptsSize] - cy + 5;

        cv::circle(img, cv::Point2f(x, y), 2, cv::Scalar(0, 0, 255), -1);
    }

    cv::imwrite(filePath, img);
}

void write_shape(uint8_t *img, int width, int height, int stride, Shape &shape, const char *filePath){
    cv::Mat sImg(height, width, CV_8UC1, img, stride);
    cv::Mat colorImg;

    cv::cvtColor(sImg, colorImg, cv::COLOR_GRAY2BGR);

    int ptsSize = shape.ptsSize;

    for(int p = 0; p < ptsSize; p++)
        cv::circle(colorImg, cv::Point2f(shape.pts[p], shape.pts[p + ptsSize]), 2, cv::Scalar(0, 0, 255), -1);

    cv::imwrite(filePath, colorImg);
}


void write_images(SampleSet *set, int step, const char *outDir){
    char command[256];
    int ret;

    int WINW = set->WINW;
    int WINH = set->WINH;

    sprintf(command, "mkdir -p %s", outDir);

    ret = system(command);

    step = HU_MAX(step, 1);

    for(int i = 0; i < set->ssize; i += step){
        char filePath[256];
        Sample *sample = set->samples[i];

        sprintf(filePath, "%s/%s.jpg", outDir, sample->patchName);

        write_shape(sample->img, WINW, WINH, WINW, sample->oriShape, filePath);
    }
}
