#include "tool.h"


int read_file_list(const char *filePath, std::vector<std::string> &fileList)
{
    char line[512];
    FILE *fin = fopen(filePath, "r");

    if(fin == NULL){
        printf("Can't open file: %s\n", filePath);
        return -1;
    }

    while(fscanf(fin, "%s\n", line) != EOF){
        fileList.push_back(line);
    }

    fclose(fin);

    return 0;
}


void analysis_file_path(const char* filePath, char *rootDir, char *fileName, char *ext)
{
    int len = strlen(filePath);
    int idx = len - 1, idx2 = 0;

    while(idx >= 0){
        if(filePath[idx] == '.')
            break;
        idx--;
    }

    if(idx >= 0){
        strcpy(ext, filePath + idx + 1);
        ext[len - idx] = '\0';
    }
    else {
        ext[0] = '\0';
        idx = len - 1;
    }

    idx2 = idx;
    while(idx2 >= 0){
#ifdef WIN32
        if(filePath[idx2] == '\\')
#else
        if(filePath[idx2] == '/')
#endif
            break;

        idx2 --;
    }

    if(idx2 > 0){
        strncpy(rootDir, filePath, idx2);
        rootDir[idx2] = '\0';
    }
    else{
        rootDir[0] = '.';
        rootDir[1] = '\0';
    }

    strncpy(fileName, filePath + idx2 + 1, idx - idx2 - 1);
    fileName[idx - idx2 - 1] = '\0';
}




#define FIX_INTER_POINT 14

void resizer_bilinear_gray(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts){
    uint16_t *xtable = NULL;

    uint16_t FIX_0_5 = 1 << (FIX_INTER_POINT - 1);
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
		xtable[x * 2 + 1] = (1 << FIX_INTER_POINT) - (xc - x0) * (1 << FIX_INTER_POINT);
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

        wy0 = (1 << FIX_INTER_POINT) - uint16_t((yc - y0) * (1 << FIX_INTER_POINT));

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

            vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
			ptrDst[0] = (wy0 * (vy0 - vy1) + (vy1 << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //2
            x0 = ptrTab[2], x1 = x0 + 1;
            wx0 = ptrTab[3];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

			vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
			vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
			ptrDst[1] = (wy0 * (vy0 - vy1) + (vy1 << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //3
            x0 = ptrTab[4], x1 = x0 + 1;
            wx0 = ptrTab[5];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

			vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
			vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
			ptrDst[2] = (wy0 * (vy0 - vy1) + (vy1 << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //4
            x0 = ptrTab[6], x1 = x0 + 1;
            wx0 = ptrTab[7];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

			vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
			vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
			ptrDst[3] = (wy0 * (vy0 - vy1) + (vy1 << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

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

            vy0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            vy1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            dst[y * dsts + x] = (wy0 * (vy0 - vy1) + (vy1 << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrTab += 2;
        }

        dId += dsts;
    }


    delete [] xtable;
}


void affine_image(uint8_t *src, int srcw, int srch, int srcs, HPoint2f cenS,
        uint8_t *dst, int dstw, int dsth, int dsts, HPoint2f cenD, float scale, float angle){

    int FIX_ONE = 1 << FIX_INTER_POINT;
    int FIX_0_5 = FIX_ONE >> 1;

    float sina = sin(-angle) / scale;
    float cosa = cos(-angle) / scale;

    int id = 0;

    int *xtable = new int[(dstw << 1) + (dsth << 1)]; assert(xtable != NULL);
    int *ytable = xtable + (dstw << 1);


    int fcx = cenS.x * FIX_ONE;
    int fcy = cenS.y * FIX_ONE;

    for(int i = 0; i < dsth; i++){
        int idx = i << 1;

        float y = (i - cenD.y);

        ytable[idx]     = y * sina * FIX_ONE + fcx;
        ytable[idx + 1] = y * cosa * FIX_ONE + fcy;
    }

    for(int i = 0; i < dstw; i++){
        int idx = i << 1;

        float x = (i - cenD.x);

        xtable[idx]     = x * sina * FIX_ONE;
        xtable[idx + 1] = x * cosa * FIX_ONE;
    }

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

            if(x0 < 0 || x0 >= srcw || y0 < 0 || y0 >= srch)
                continue;

            assert(wx <= FIX_ONE && wy <= FIX_ONE);

            uint8_t *ptr1 = src + y0 * srcs + x0;
            uint8_t *ptr2 = ptr1 + srcs;

            uint8_t value0 = ((ptr1[0] << FIX_INTER_POINT) + (ptr1[1] - ptr1[0]) * wx + FIX_0_5) >> FIX_INTER_POINT;
            uint8_t value1 = ((ptr2[0] << FIX_INTER_POINT) + (ptr2[1] - ptr2[0]) * wx + FIX_0_5) >> FIX_INTER_POINT;

            dst[id + x] = ((value0 << FIX_INTER_POINT) + (value1 - value0) * wy + FIX_0_5) >> FIX_INTER_POINT;
        }

        id += dsts;
    }

    delete [] xtable;
}


void extract_area_from_image(uint8_t *img, int width, int height, int stride, uint8_t *patch, HRect &rect){
    int x0 = rect.x;
    int y0 = rect.y;

    int x1 = x0 + rect.width - 1;
    int y1 = y0 + rect.height - 1;

    int faceSize = HU_MAX(rect.width, rect.height);

    int w, h;

    int bl, bt, br, bb;

    memset(patch, 0, sizeof(uint8_t) * faceSize * faceSize);

    bl = 0, bt = 0, br = 0, bb = 0;

    if(x0 < 0) {
        bl = -x0;
        x0 = 0;
    }

    if(y0 < 0){
        bt = -y0;
        y0 = 0;
    }

    if(x1 > width - 1){
        br = x1 - width + 1;
        x1 = width - 1;
    }

    if(y1 > height - 1){
        bb = y1 - height + 1;
        y1 = height - 1;
    }

    w = faceSize - bl - br;
    h = faceSize - bt - bb;

    patch += bt * faceSize + bl;
    img += y0 * stride + x0;

    for(int y = 0; y < h; y++){
        memcpy(patch, img, sizeof(uint8_t) * w);

        patch += faceSize;
        img += stride;
    }

    rect.x = x0 - bl;
    rect.y = y0 - bt;
}


void normalize_feature(float *feats, int ssize, int featDim, int step, float *vmean, float *vstd){
    double *sumv = new double[featDim];

    memset(sumv, 0, sizeof(double) * featDim);

    float *ptrFeats = feats;

    for(int i = 0; i < ssize; i++){
        for(int j = 0; j < featDim; j++)
            sumv[j] += ptrFeats[j];

        ptrFeats += step;
    }

    for(int j = 0; j < featDim; j++)
        vmean[j] = sumv[j] / ssize;


    memset(sumv, 0, sizeof(double) * featDim);

    ptrFeats = feats;
    for(int i = 0; i < ssize; i++){
        for(int j = 0; j < featDim; j++){
            float t = ptrFeats[j] - vmean[j];
            ptrFeats[j] = t;

            sumv[j] += t * t;
        }

        ptrFeats += step;
    }

    for(int j = 0; j < featDim; j++)
        vstd[j] = sqrtf(sumv[j] / ssize);

    ptrFeats = feats;
    for(int i = 0; i < ssize; i++){
        for(int j = 0; j < featDim; j++)
            ptrFeats[j] /= vstd[j];

        ptrFeats += step;
    }

    delete [] sumv;
    sumv = NULL;
}


void transform_image(uint8_t *img, int width, int height, int stride){
    static cv::RNG rng0(cv::getTickCount());
    static cv::RNG rng1(rng0.uniform(0, 10000));
    static cv::RNG rng2(rng1.uniform(0, 10000));
    static cv::RNG rng3(rng2.uniform(0, 10000));

    if(rng0.uniform(0, 10) == 0){
        uint8_t *data = img;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++)
                if(data[x] > 15 && data[x] < 240)
                    data[x] += rng2.uniform(-8, 9);

            data += stride;
        }
    }

    cv::Mat src(height, width, CV_8UC1, img, stride);

    if(rng1.uniform(0, 10) == 0){
        int k = 2 * rng1.uniform(1, 4) + 1;

        cv::Mat blur;
        cv::GaussianBlur(src, blur, cv::Size(k , k), 0, 0);

        blur.copyTo(src);
    }

    if(rng2.uniform(0, 20) == 1){
        cv::Mat res;
        cv::equalizeHist(src, res);
        res.copyTo(src);
    }

    if(rng3.uniform(0, 20) == 0){
        double sum = 0;

        uint8_t *data = img;
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++)
                sum += data[x];
            data += stride;
        }

        sum /= (width * height);
        sum += 0.1;

        data = img;
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                float t = (data[x] - sum) / (data[x] + sum) + 1.0f;

                data[x] = uint8_t(t * 0.5f * 255);
            }

            data += stride;
        }
    }
}


void write_matrix(float *data, int cols, int rows, int step, const char *outfile){
    FILE *fout = fopen(outfile, "w");

    if(fout == NULL){
        printf("Can't open file %s\n", outfile);
        return;
    }

    step = HU_MAX(step, 1);
    for(int y = 0; y < rows; y += step){
        float *ptr = data + y * cols;
        for(int x = 0; x < cols; x++)
            fprintf(fout, "%10.6f ", ptr[x]);

        fprintf(fout, "\n");
    }

    fclose(fout);
}


#define Q 14

void bgra2gray(uint8_t *bgraData, int width, int height, int stride, uint8_t *data)
{
    const short SCALE_B = ((short)(0.1140f * (1 << Q)));
    const short SCALE_G = ((short)(0.5870f * (1 << Q)));
    const short SCALE_R = ((short)(0.2989f * (1 << Q)));

    const short DELTA =  ((short)(1 << (Q - 1)));

    uint8_t *ptrRes;
    uint8_t *ptrSrc;

    int x, y;

    for(y = 0; y <= height - 4; y += 4){
        //0
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_B + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_R + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_B + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_R + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_B + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_R + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_B + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_R + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_B + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_R + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //1
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_B + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_R + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_B + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_R + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_B + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_R + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_B + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_R + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_B + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_R + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //2
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_B + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_R + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_B + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_R + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_B + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_R + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_B + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_R + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_B + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_R + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //3
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_B + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_R + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_B + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_R + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_B + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_R + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_B + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_R + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_B + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_R + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;
    }

    for(; y < height; y++){
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_B + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_R + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_B + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_R + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_B + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_R + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_B + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_R + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_B + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_R + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;
    }
}



void rgba2gray(uint8_t *bgraData, int width, int height, int stride, uint8_t *data)
{
    const short SCALE_B = ((short)(0.1140f * (1 << Q)));
    const short SCALE_G = ((short)(0.5870f * (1 << Q)));
    const short SCALE_R = ((short)(0.2989f * (1 << Q)));

    const short DELTA =  ((short)(1 << (Q - 1)));

    uint8_t *ptrRes;
    uint8_t *ptrSrc;

    int x, y;

    for(y = 0; y <= height - 4; y += 4){
        //0
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_R + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_B + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_R + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_B + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_R + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_B + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_R + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_B + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_R + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_B + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //1
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_R + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_B + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_R + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_B + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_R + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_B + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_R + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_B + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_R + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_B + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //2
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_R + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_B + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_R + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_B + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_R + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_B + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_R + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_B + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_R + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_B + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;

        //3
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_R + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_B + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_R + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_B + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_R + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_B + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_R + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_B + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_R + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_B + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;
    }

    for(; y < height; y++){
        ptrRes = data;
        ptrSrc = bgraData;

        for(x = 0; x <= width - 4; x += 4, ptrSrc += 16, ptrRes += 4){
            ptrRes[0] = (ptrSrc[0]  * SCALE_R + ptrSrc[1]  * SCALE_G + ptrSrc[2]  * SCALE_B + DELTA) >> Q;
            ptrRes[1] = (ptrSrc[4]  * SCALE_R + ptrSrc[5]  * SCALE_G + ptrSrc[6]  * SCALE_B + DELTA) >> Q;
            ptrRes[2] = (ptrSrc[8]  * SCALE_R + ptrSrc[9]  * SCALE_G + ptrSrc[10] * SCALE_B + DELTA) >> Q;
            ptrRes[3] = (ptrSrc[12] * SCALE_R + ptrSrc[13] * SCALE_G + ptrSrc[14] * SCALE_B + DELTA) >> Q;
        }

        for(; x < width; x ++, ptrRes ++, ptrSrc += 4){
            ptrRes[0] = (ptrSrc[0] * SCALE_R + ptrSrc[1] * SCALE_G + ptrSrc[2] * SCALE_B + DELTA) >> Q;
        }

        data += width;
        bgraData += stride;
    }
}


static void transpose_wx8_c_code_channel_one(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride, int width)
{
    int i;

    for (i = 0; i < width; i++) {
        dst[0] = src[0 * src_stride];
        dst[1] = src[1 * src_stride];
        dst[2] = src[2 * src_stride];
        dst[3] = src[3 * src_stride];
        dst[4] = src[4 * src_stride];
        dst[5] = src[5 * src_stride];
        dst[6] = src[6 * src_stride];
        dst[7] = src[7 * src_stride];

        src++;
        dst += dst_stride;
    }
}


static void transpose_wx8_c_code_channel_three(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride, int width)
{
    int i;
    for (i = 0; i < width; i++) {
        dst[0] = src[0 * src_stride + 0];
        dst[1] = src[0 * src_stride + 1];
        dst[2] = src[0 * src_stride + 2];

        dst[3] = src[1 * src_stride + 0];
        dst[4] = src[1 * src_stride + 1];
        dst[5] = src[1 * src_stride + 2];

        dst[6] = src[2 * src_stride + 0];
        dst[7] = src[2 * src_stride + 1];
        dst[8] = src[2 * src_stride + 2];

        dst[9] = src[3 * src_stride + 0];
        dst[10] = src[3 * src_stride + 1];
        dst[11] = src[3 * src_stride + 2];

        dst[12] = src[4 * src_stride + 0];
        dst[13] = src[4 * src_stride + 1];
        dst[14] = src[4 * src_stride + 2];

        dst[15] = src[5 * src_stride + 0];
        dst[16] = src[5 * src_stride + 1];
        dst[17] = src[5 * src_stride + 2];

        dst[18] = src[6 * src_stride + 0];
        dst[19] = src[6 * src_stride + 1];
        dst[20] = src[6 * src_stride + 2];

        dst[21] = src[7 * src_stride + 0];
        dst[22] = src[7 * src_stride + 1];
        dst[23] = src[7 * src_stride + 2];

        src += 3;
        dst += dst_stride;
    }

}


static void transpose_wxh_c_code(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride,
        int width, int height, int channels)
{
    int i,j;
    if (channels == 1){
        for (i = 0; i < width; i++) {
            j = 0;
            for (; j <= height - 4; j += 4) {
                dst[i * dst_stride + j + 0] = src[(j + 0) * src_stride + i];
                dst[i * dst_stride + j + 1] = src[(j + 1) * src_stride + i];
                dst[i * dst_stride + j + 2] = src[(j + 2) * src_stride + i];
                dst[i * dst_stride + j + 3] = src[(j + 3) * src_stride + i];
            }

            for (; j < height; ++j) {
                dst[i * dst_stride + j] = src[j * src_stride + i];
            }
        }
    } else if (channels == 3){
        for (i = 0; i < width; i++) {
            j = 0;
            for (; j <= height - 4; j += 4) {
                dst[i * dst_stride + j + 0] = src[(j + 0) * src_stride + 3 * i + 0];
                dst[i * dst_stride + j + 1] = src[(j + 0) * src_stride + 3 * i + 1];
                dst[i * dst_stride + j + 2] = src[(j + 0) * src_stride + 3 * i + 2];

                dst[i * dst_stride + j + 3] = src[(j + 1) * src_stride + 3 * i + 0];
                dst[i * dst_stride + j + 4] = src[(j + 1) * src_stride + 3 * i + 1];
                dst[i * dst_stride + j + 5] = src[(j + 1) * src_stride + 3 * i + 2];

                dst[i * dst_stride + j + 6] = src[(j + 2) * src_stride + 3 * i + 0];
                dst[i * dst_stride + j + 7] = src[(j + 2) * src_stride + 3 * i + 1];
                dst[i * dst_stride + j + 8] = src[(j + 2) * src_stride + 3 * i + 2];

                dst[i * dst_stride + j + 9] = src[(j + 3) * src_stride + 3 * i + 0];
                dst[i * dst_stride + j + 10] = src[(j + 3) * src_stride + 3 * i + 1];
                dst[i * dst_stride + j + 11] = src[(j + 3) * src_stride + 3 * i + 2];
            }
            for (; j < height; ++j) {
                dst[i * dst_stride + j + 0] = src[j * src_stride + 3 * i + 0];
                dst[i * dst_stride + j + 1] = src[j * src_stride + 3 * i + 1];
                dst[i * dst_stride + j + 2] = src[j * src_stride + 3 * i + 2];
            }
        }
    }  else {
    }
}



static void transpose_plane(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride,
        int width, int height, int channels)
{
    int i = height;

    if (channels == 1){

        while (i >= 8) {
            transpose_wx8_c_code_channel_one(src, src_stride, dst, dst_stride, width);
            src += 8 * src_stride;
            dst += 8;
            i -= 8;
        }

        if (i > 0) {
            transpose_wxh_c_code(src, src_stride, dst, dst_stride, width, i, 1);
        }

    } else if (channels == 3){
        while (i >= 8) {
            transpose_wx8_c_code_channel_three(src, src_stride, dst, dst_stride, width);
            src += 8 * src_stride;
            dst += 8 * 3;
            i -= 8;
        }

        if (i > 0) {
            transpose_wxh_c_code(src, src_stride, dst, dst_stride, width, i, 3);
        }
    }
}



static void rotation_plane_90(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride,
        int width, int height, int channels)
{
    src       += src_stride * (height - 1);
    src_stride = -src_stride;
    transpose_plane(src, src_stride, dst, dst_stride, width, height, channels);
}


static void rotation_plane_270(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride,
        int width, int height, int channels)
{
    dst       += dst_stride * (width - 1);
    dst_stride = -dst_stride;
    transpose_plane(src, src_stride, dst, dst_stride, width, height, channels);
}


static void mirror_copy_row_c_code(const uint8_t* src, uint8_t* dst, int width)
{
    int i;
    for (i = 0; i < width; i++){
        dst[i] = src[width - 1 - i];
    }
}


static void rotate_plane_180(const uint8_t* src, int src_stride,
        uint8_t* dst, int dst_stride,
        int width, int height, int channels)
{

    uint8_t* row           = (uint8_t*)malloc(width);
    const uint8_t* src_bot = src + src_stride * (height - 1);
    uint8_t* dst_bot       = dst + dst_stride * (height - 1);
    int half_height     = (height + 1) >> 1;
    int y;
    void(*ptr_mirror_copy_row)(const uint8_t* src, uint8_t* dst, int width) = mirror_copy_row_c_code;

    for (y = 0; y < half_height; ++y) {
        ptr_mirror_copy_row(src, row, width);
        src += src_stride;
        ptr_mirror_copy_row(src_bot, dst, width);
        dst += dst_stride;
        memcpy(dst_bot, row, width);
        src_bot -= src_stride;
        dst_bot -= dst_stride;
    }
    free(row);
}


static void copy_plane(uint8_t* src, int src_stride, uint8_t* dst, int dst_stride, int width, int height)
{
    int y;
    for (y = 0; y < height; y++) {
        memcpy(dst, src, width);
        src += src_stride;
        dst += dst_stride;
    }
}


void rotate_width_degree(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int &dstw, int &dsth, int &dsts, int degree)
{

    if(degree == 0){
        memcpy(dst, src, sizeof(uint8_t) * srcs * srch);
        dstw = srcw;
        dsth = srch;
        dsts = srcs;
    }
    else if(degree == 1){
        dstw = srch;
        dsth = srcw;
        dsts = srch;

        rotation_plane_90(src, srcs, dst, dsts, srcw, srch, 1);
    }
    else if(degree == 2){
        dstw = srcw;
        dsth = srch;
        dsts = srcs;
        rotate_plane_180(src, srcs, dst, dsts, srcw, srch, 1);
    }
    else if(degree == 3){
        dstw = srch;
        dsth = srcw;
        dsts = srch;

        rotation_plane_270(src, srcs, dst, dsts, srcw, srch, 1);
    }
}
