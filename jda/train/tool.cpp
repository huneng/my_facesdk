#include "tool.h"


int read_file_list(const char *filePath, std::vector<std::string> &fileList)
{
    char line[512];
    FILE *fin = fopen(filePath, "r");

    if(fin == NULL){
        printf("Can't open file: %s\n", filePath);
        return -1;
    }

    while(fgets(line, 511, fin) != NULL){
        line[strlen(line) - 1] = '\0';
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
    uint16_t *table = NULL;

    uint16_t FIX_0_5 = 1 << (FIX_INTER_POINT - 1);
    float scalex, scaley;

    scalex = srcw / float(dstw);
    scaley = srch / float(dsth);

    table = new uint16_t[dstw * 3];

    for(int i = 0; i < dstw; i++){
        float x = i * scalex;

        if(x < 0) x = 0;
        if(x >= srcw - 1) x = srcw - 1.01f;

        int x0 = int(x);

        table[i * 3] = x0;
        table[i * 3 + 2] = (x - x0) * (1 << FIX_INTER_POINT);
        table[i * 3 + 1] = (1 << FIX_INTER_POINT) - table[i * 3 + 2];
    }

    int sId = 0, dId = 0;

    for(int y = 0; y < dsth; y++){
        int x;
        float yc;

        uint16_t wy0, wy1;
        uint16_t y0, y1;
        uint16_t *ptrTab = table;
        int buffer[8];

        yc = y * scaley;

        if(yc < 0) yc = 0;
        if(yc >= srch - 1) yc = srch - 1.01f;

        y0 = uint16_t(yc);
        y1 = y0 + 1;

        wy1 = uint16_t((yc - y0) * (1 << FIX_INTER_POINT));
        wy0 = (1 << FIX_INTER_POINT) - wy1;

        sId = y0 * srcs;

        uint8_t *ptrDst = dst + dId;

        for(x = 0; x <= dstw - 4; x += 4){
            uint16_t x0, x1, wx0, wx1;
            uint8_t *ptrSrc0, *ptrSrc1;

            //1
            x0 = ptrTab[0], x1 = x0 + 1;
            wx0 = ptrTab[1], wx1 = ptrTab[2];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[0] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[1] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //2
            x0 = ptrTab[3], x1 = x0 + 1;

            wx0 = ptrTab[4], wx1 = ptrTab[5];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[2] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[3] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //3
            x0 = ptrTab[6], x1 = x0 + 1;

            wx0 = ptrTab[7], wx1 = ptrTab[8];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[4] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[5] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            //4
            x0 = ptrTab[9], x1 = x0 + 1;
            wx0 = ptrTab[10], wx1 = ptrTab[11];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            buffer[6] = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            buffer[7] = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrDst[0] = (wy0 * (buffer[0] - buffer[1]) + (buffer[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[1] = (wy0 * (buffer[2] - buffer[3]) + (buffer[3] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[2] = (wy0 * (buffer[4] - buffer[5]) + (buffer[5] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            ptrDst[3] = (wy0 * (buffer[6] - buffer[7]) + (buffer[7] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrDst += 4;
            ptrTab += 12;
        }

        for(; x < dstw; x++){
            uint16_t x0, x1, wx0, wx1, valuex0, valuex1;

            uint8_t *ptrSrc0, *ptrSrc1;
            x0 = ptrTab[0], x1 = x0 + 1;

            wx0 = ptrTab[1], wx1 = ptrTab[2];

            ptrSrc0 = src + sId + x0;
            ptrSrc1 = src + sId + srcs + x0;

            valuex0 = (wx0 * (ptrSrc0[0] - ptrSrc0[1]) + (ptrSrc0[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;
            valuex1 = (wx0 * (ptrSrc1[0] - ptrSrc1[1]) + (ptrSrc1[1] << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            dst[y * dsts + x] = (wy0 * (valuex0 - valuex1) + (valuex1 << FIX_INTER_POINT) + FIX_0_5) >> FIX_INTER_POINT;

            ptrTab += 3;
        }

        dId += dsts;
    }


    delete [] table;
}


#define LT(a, b) (a < b)
IMPLEMENT_QSORT(sort_arr_float, float, LT);

void affine_image(uint8_t *src, int srcw, int srch, int srcs, float angle, float scale, cv::Point2f &center,
        uint8_t *dst, int dstw, int dsth, int dsts){

    int FIX_ONE = 1 << FIX_INTER_POINT;
    int FIX_0_5 = FIX_ONE >> 1;

    float sina = sin(-angle) / scale;
    float cosa = cos(-angle) / scale;

    int id = 0;

    int *xtable = new int[(dstw << 1) + (dsth << 1)]; assert(xtable != NULL);
    int *ytable = xtable + (dstw << 1);

    float cx = (float)dstw / 2;
    float cy = (float)dsth / 2;

    int fcx = center.x * FIX_ONE;
    int fcy = center.y * FIX_ONE;

    for(int i = 0; i < dsth; i++){
        int idx = i << 1;

        float y = (i - cy);

        ytable[idx]     = y * sina * FIX_ONE + fcx;
        ytable[idx + 1] = y * cosa * FIX_ONE + fcy;
    }

    for(int i = 0; i < dstw; i++){
        int idx = i << 1;

        float x = (i - cx);

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


void mean_filter_3x3(uint8_t *img, int width, int height, int stride){
    uint8_t *buffer = new uint8_t[stride * height];

    memcpy(buffer, img, sizeof(uint8_t) * stride * height);

    for(int y = 1; y < height - 1; y++){
        for(int x = 1; x < width - 1; x++){
            int id = y * stride + x;

            uint8_t *ptr = img + id;

            uint32_t value = ptr[-stride - 1] + ptr[-stride] + ptr[-stride + 1] +
                             ptr[-1] + ptr[0] + ptr[1] +
                             ptr[ stride - 1] + ptr[ stride] + ptr[stride + 1];

            buffer[id] = value * 0.111111f;
        }
    }

    memcpy(img, buffer, sizeof(uint8_t) * stride * height);

    delete [] buffer;
}


uint8_t* mean_filter_3x3_res(uint8_t *img, int width, int height, int stride){

    uint8_t *buffer = new uint8_t[width * height];

    for(int y = 1; y < height - 1; y++){
        for(int x = 1; x < width - 1; x++){
            uint8_t *ptr = img + y * stride + x;

            uint32_t value = ptr[-stride - 1] + ptr[-stride] + ptr[-stride + 1] +
                             ptr[-1] + ptr[0] + ptr[1] +
                             ptr[ stride - 1] + ptr[ stride] + ptr[stride + 1];

            buffer[y * width + x] = value * 0.111111f;
        }
    }

    return buffer;
}


void mean_filter_3x3(uint8_t *img, int width, int height, int stride, uint8_t *res){
    for(int y = 1; y < height - 1; y++){
        for(int x = 1; x < width - 1; x++){

            uint8_t *ptr = img + y * stride + x;

            uint32_t value = ptr[-stride - 1] + ptr[-stride] + ptr[-stride + 1] +
                ptr[-1] + ptr[0] + ptr[1] +
                ptr[ stride - 1] + ptr[ stride] + ptr[stride + 1];

            res[y * width + x] = value * 0.111111f;
        }
    }
}


void vertical_mirror(uint8_t *img, int width, int height, int stride)
{
    int cy = height / 2;

    for(int y = 0; y < cy; y++){
        uint8_t *ptr1 = img + y * stride;
        uint8_t *ptr2 = img + (height - y - 1) * stride;

        for(int x = 0; x < width; x++)
            HU_SWAP(ptr1[x], ptr2[x], uint8_t);
    }
}


void horizontal_mirror(uint8_t *img, int width, int height, int stride){
    int cx = width / 2;
    uint8_t *ptr = img;

    for(int y = 0; y < height; y++){
        for(int x = 0; x < cx; x++)
            HU_SWAP(ptr[x], ptr[width - x - 1], uint8_t);
        ptr += stride;
    }
}


void transform_image(uint8_t *img, int width, int height, int stride){
    static cv::RNG rng(cv::getTickCount());

    assert(stride * height < 4096 * 4096);

    if(rng.uniform(0, 10) == 0){
        uint8_t *data = img;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                if(data[x] > 16 && data[x] < 239)
                    data[x] += rng.uniform(-8, 8);
            }

            data += stride;
        }
    }

    if(rng.uniform(0, 10) == 1){
        cv::Mat sImg(height, width, CV_8UC1, img);
        cv::Mat blur;

        cv::GaussianBlur(sImg, blur, cv::Size(5, 5), 0, 0);

        for(int y = 0; y < height; y++)
            memcpy(img + y * stride, blur.data + y * blur.step, sizeof(uint8_t) * width);
    }
}


void transform_image(uint8_t *img, int width, int height, int stride, uint8_t *dImg){
    static cv::RNG rng(cv::getTickCount());

    float angle = rng.uniform(-HU_PI, HU_PI);
    float scale = rng.uniform(0.7, 1.3);

    assert(stride * height < 4096 * 4096);

    cv::Point2f center(width >> 1, height >> 1);

    memset(dImg, 0, sizeof(uint8_t) * height * stride);

    affine_image(img, width, height, stride, angle, scale, center, dImg, width, height, stride);

    memcpy(img, dImg, height * stride);

    if(rng.uniform(0, 8) == 0){
        uint8_t *data = img;

        float sum = 0;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++)
                sum += data[x];

            data += stride;
        }

        sum /= (width * height);
        sum ++;

        data = img;
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++)
                data[x] = ((data[x] - sum) / (data[x] + sum) + 1.0f) * 0.5 * 255;

            data += stride;
        }
    }

    if(rng.uniform(0, 8) == 0){
        uint8_t *data = img;

        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                if(data[x] > 16 && data[x] < 239)
                    data[x] += rng.uniform(-8, 8);
            }

            data += stride;
        }
    }

    if(rng.uniform(0, 8) == 0)
        vertical_mirror(img, width, height, stride);

    if(rng.uniform(0, 8) == 0)
        horizontal_mirror(img, width, height, stride);

    if(rng.uniform(0, 8) == 0){
        cv::Mat sImg(height, width, CV_8UC1, img);
        cv::Mat blur;

        cv::GaussianBlur(sImg, blur, cv::Size(5, 5), 0, 0);

        for(int y = 0; y < height; y++)
            memcpy(img + y * stride, blur.data + y * blur.step, sizeof(uint8_t) * width);
    }


    if(rng.uniform(0, 8) == 0){
        cv::Mat sImg(height, width, CV_8UC1, img);
        cv::Mat blur;

        cv::equalizeHist(sImg, blur);

        for(int y = 0; y < height; y++)
            memcpy(img + y * stride, blur.data + y * blur.step, sizeof(uint8_t) * width);
    }
}



void write_matrix(float *data, int cols, int rows, const char *outFile){
    FILE *fout = fopen(outFile, "w");
    if(fout == NULL)
        return ;

    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++)
            fprintf(fout, "%f ", data[x]);
        fprintf(fout, "\n");
        data += cols;
    }

    fclose(fout);
}


void write_matrix(double *data, int cols, int rows, const char *outFile){
    FILE *fout = fopen(outFile, "w");
    if(fout == NULL)
        return ;

    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++)
            fprintf(fout, "%f ", data[x]);
        fprintf(fout, "\n");
        data += cols;
    }

    fclose(fout);
}


void write_matrix(float **matrix, int cols, int rows, const char *outFile){
    FILE *fout = fopen(outFile, "w");
    if(fout == NULL)
        return ;

    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++)
            fprintf(fout, "%f ", matrix[y][x]);
        fprintf(fout, "\n");
    }

    fclose(fout);
}
