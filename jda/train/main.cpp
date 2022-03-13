#include "cascade.h"
#include <sys/time.h>

void extract_face_from_image(cv::Mat &src, cv::Rect &rect, cv::Mat &patch, Shape &shape);
void rotate_with_degree(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int &dstw, int &dsth, int &dsts, int degree);

int main_train(int argc, char **argv){
    if(argc < 4){
        printf("Usage: %s [pos list] [neg list] [out model]\n", argv[0]);
        return 1;
    }

    JDADetector *detector = NULL;

    train(&detector, argv[1], argv[2], 80, 80, 5);

    if(detector == NULL)
        return 2;

    save(argv[3], detector);

    release(&detector);

    return 0;
}


int main_detect_video(int argc, char **argv){
    if(argc < 3){
        printf("Usage: %s [model] [video]\n", argv[0]);
        return 1;
    }

    JDADetector *detector = NULL;

    if(load(argv[1], &detector) != 0){
        printf("Load model %s error\n", argv[1]);
        return 2;
    }

    struct timezone tz;
    struct timeval stv, etv;

    set_detect_factor(detector, 0.1, 1.0, 0.1, 0.1, 12);

    cv::VideoCapture cap(argv[2]);

    if(!cap.isOpened()){
        printf("Can't open video %s\n", argv[2]);
        return 3;
    }

    long totalFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);

    for(int fid = 0; fid < totalFrame; fid ++){
        cv::Mat frame;
        cv::Mat gray;

        cap >> frame;
        if(frame.empty()) continue;

        Shape *shapes = NULL;
        HRect *rects = NULL;
        float *confs = NULL;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        gettimeofday(&stv, &tz);
        int count = detect(detector, gray.data, gray.cols, gray.rows, gray.step, &rects, &shapes, &confs);
        gettimeofday(&etv, &tz);

        printf("%.2f ms\n", 1000 * (etv.tv_sec - stv.tv_sec) + 0.001f * (etv.tv_usec - stv.tv_usec));

        if(count > 0){
            for(int i = 0; i < count; i++){
                HRect rect = rects[i];
                HPoint2f *pts = shapes[i].pts;
                int ptsSize = shapes[i].ptsSize;

                //printf("%d %d %d %d\n", rect.x, rect.y, rect.width, rect.height);
                cv::rectangle(frame, cv::Rect(rect.x, rect.y, rect.width, rect.height), cv::Scalar(255, 0, 0), 2);

                for(int p = 0; p < ptsSize; p++)
                    cv::circle(frame, cv::Point2f(pts[p].x, pts[p].y), 3, cv::Scalar(0, 255, 0), -1);
            }

            delete [] shapes;
            delete [] rects;
            delete [] confs;
        }

        cv::imshow("frame", frame);
        cv::waitKey(1);
    }


    release(&detector);
    return 0;
}


HRect rotate_rect_reverse(HRect rect, int width, int height, int degree){
    int cx = width >> 1;
    int cy = height >> 1;

    int x0, y0, x1, y1;
    int rx0, ry0, rx1, ry1;
    x0 = rect.x;
    y0 = rect.y;
    x1 = rect.x + rect.width;
    y1 = rect.y + rect.height;

    switch(degree){
        case 0:
            break;

        case 1:
            rx0 = y0;
            ry0 = width - 1 - x1;
            rx1 = y1;
            ry1 = width - 1 - x0;
            break;

        case 2:
            rx0 = width - 1 - x1;
            ry0 = height - 1 - y1;
            rx1 = width - 1 - x0;
            ry1 = height - 1 - y0;
            break;

        case 3:
            rx0 = height - 1 - y1;
            ry0 = x0;
            rx1 = height - 1 - y0;
            ry1 = x1;
            break;
    }

    rect.x = rx0;
    rect.y = ry0;
    rect.width = rx1 - rx0 + 1;
    rect.height = ry1 - ry0 + 1;

    return rect;
}


int main_generate_samples(int argc, char **argv)
{
    if(argc < 4){
        printf("Usage: %s [model] [image list] [out dir]\n", argv[0]);
        return 1;
    }

    JDADetector *detector;
    int ret, size, count;
    std::vector<std::string> imageList;

    if(load(argv[1], &detector) != 0){
        printf("Load model %s error\n", argv[1]);
        return 2;
    }

    set_detect_factor(detector, 0.1, 1.0, 0.1, 0.1, 15);

    read_file_list(argv[2], imageList);

    size = imageList.size();

    count = 0;
//#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < size; i++){
        char rootDir[128], fileName[128], ext[30], outPath[256];
        const char *imgPath = imageList[i].c_str();
        cv::Mat img = cv::imread(imgPath, 1);

        printf("%d %s\n", i, imgPath);
        HRect *rects = new HRect[200];
        int rsize = 0;

        if(img.empty()) continue;

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        {
            Shape *rshapes = NULL;
            HRect *rrects = NULL;
            float *rconfs = NULL;
            int rrsize;

            rrsize = detect(detector, gray.data, gray.cols, gray.rows, gray.step, &rrects, &rshapes, &rconfs);

            if(rrsize > 0){
                memcpy(rects + rsize, rrects, sizeof(HRect) * rrsize);
                rsize += rrsize;

                delete [] rrects;
                delete [] rshapes;
                delete [] rconfs;
            }

            /*
            uint8_t *dst = new uint8_t[gray.step * gray.rows];
            int dstw, dsth, dsts;

            rotate_with_degree(gray.data, gray.cols, gray.rows, gray.step, dst, dstw, dsth, dsts, 1);

            rrsize = detect(detector, dst, dstw, dsth, dsts, &rrects, &rshapes, &rconfs);

            if(rrsize > 0){
                for(int j = 0; j < rrsize; j++)
                    rrects[j] = rotate_rect_reverse(rrects[j], dstw, dsth, 1);

                memcpy(rects + rsize, rrects, sizeof(HRect) * rrsize);

                rsize += rrsize;

                delete [] rrects;
                delete [] rshapes;
                delete [] rconfs;
            }

            rotate_with_degree(gray.data, gray.cols, gray.rows, gray.step, dst, dstw, dsth, dsts, 2);

            rrsize = detect(detector, dst, dstw, dsth, dsts, &rrects, &rshapes, &rconfs);

            if(rrsize > 0){
                for(int j = 0; j < rrsize; j++)
                    rrects[j] = rotate_rect_reverse(rrects[j], dstw, dsth, 2);

                memcpy(rects + rsize, rrects, sizeof(HRect) * rrsize);

                rsize += rrsize;

                delete [] rrects;
                delete [] rshapes;
                delete [] rconfs;
            }

            rotate_with_degree(gray.data, gray.cols, gray.rows, gray.step, dst, dstw, dsth, dsts, 3);

            rrsize = detect(detector, dst, dstw, dsth, dsts, &rrects, &rshapes, &rconfs);

            if(rrsize > 0){
                for(int j = 0; j < rrsize; j++)
                    rrects[j] = rotate_rect_reverse(rrects[j], dstw, dsth, 3);

                memcpy(rects + rsize, rrects, sizeof(HRect) * rrsize);

                rsize += rrsize;

                delete [] rrects;
                delete [] rshapes;
                delete [] rconfs;
            }

            delete [] dst;
            //*/
        }

        analysis_file_path(imgPath, rootDir, fileName, ext);

        for(int j = 0; j < rsize; j++){
            HRect rect = rects[j];

            cv::Mat patch;
            cv::Rect crect;

            crect.x = HU_MAX(0, rect.x);
            crect.y = HU_MAX(0, rect.y);

            crect.width = HU_MIN(crect.x + rect.width, gray.cols) - crect.x;
            crect.height = HU_MIN(crect.y + rect.height, gray.rows) - crect.y;

            patch = cv::Mat(img, crect);

            cv::resize(patch, patch, cv::Size(96, 96));
            sprintf(outPath, "%s/%s_%d.jpg", argv[3], fileName, j);
            cv::imwrite(outPath, patch);
        }

#pragma omp critical
        {
            count++;
            //printf("%d\r", count), fflush(stdout);
        }
        if(rsize > 0) delete [] rects;
    }

    release(&detector);

    return 0;
}


int main_detect_images(int argc, char **argv)
{
    if(argc < 4){
        printf("Usage: %s [model] [image list] [out dir]\n", argv[0]);
        return 1;
    }

    JDADetector *detector;
    int ret, size;
    std::vector<std::string> imageList;

    if(load(argv[1], &detector) != 0){
        printf("Load model %s error\n", argv[1]);
        return 2;
    }

    set_detect_factor(detector, 0.1, 1.0, 0.1, 0.1, 10);

    read_file_list(argv[2], imageList);

    size = imageList.size();

    struct timezone tz;
    struct timeval stv, etv;

    for(int i = 0; i < size; i++){
        char rootDir[256], fileName[256], ext[20], outPath[512];
        const char *imgPath = imageList[i].c_str();
        cv::Mat img = cv::imread(imgPath, 1);

        Shape *shapes = NULL;
        HRect *rects = NULL;
        float *confs = NULL;
        int rsize = 0;

        assert(!img.empty());

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        gettimeofday(&stv, &tz);
        rsize = detect(detector, gray.data, gray.cols, gray.rows, gray.step, &rects, &shapes, &confs);
        gettimeofday(&etv, &tz);

        printf("%.2f ms\n", 1000 * (etv.tv_sec - stv.tv_sec) + 0.001f * (etv.tv_usec - stv.tv_usec));

        analysis_file_path(imgPath, rootDir, fileName, ext);

        for(int j = 0; j < rsize; j++){
            int ptsSize = shapes[j].ptsSize;
            HRect rect = rects[j];

            cv::rectangle(img, cv::Rect(rect.x, rect.y, rect.width, rect.height), cv::Scalar(255, 0, 0), 2);
            //*
            for(int k = 0; k < ptsSize; k ++)
                cv::circle(img, cv::Point2f(shapes[j].pts[k].x, shapes[j].pts[k].y), 3, cv::Scalar(0,255, 0), -1);
            // */
        }

        if(rsize > 0){
            delete [] rects;
            delete [] shapes;
            delete [] confs;
        }

        //sprintf(outPath, "%s/%d.jpg", argv[3], i);
        sprintf(outPath, "%s/%s.jpg", argv[3], fileName);
        cv::imwrite(outPath, img);
    }

    release(&detector);

    return 0;
}


int main_detect_images_fddb(int argc, char **argv)
{
    if(argc < 5){
        printf("Usage: %s [model] [image dir] [image list] [out dir]\n", argv[0]);
        return 1;
    }

    JDADetector *detector;
    if(load(argv[1], &detector) != 0){
        printf("Load model %s error\n", argv[1]);
        return 2;
    }

    std::vector<std::string> imageList;
    int size;
    char filePath[256], rootDir[128], fileName[128], ext[30];
    FILE *fout = NULL;

    analysis_file_path(argv[3], rootDir, fileName, ext);

    read_file_list(argv[3], imageList);
    size = imageList.size();

    sprintf(filePath, "%s/%s.%s", argv[4], fileName, ext);
    fout = fopen(filePath, "w");

    set_detect_factor(detector, 0.1, 1.0, 0.2, 0.1, 12);
    for(int i = 0; i < size; i++){
        std::string imgPath = std::string(argv[2]) + "/" + imageList[i] + ".jpg";
        cv::Mat src = cv::imread(imgPath);

        if(src.empty()){
            printf("Can't open image %s\n", imgPath.c_str());
            break;
        }

        cv::Mat img(src.rows * 1.4, src.cols * 1.4, src.type(), cv::Scalar::all(0));

        int x_ = src.cols * 0.2;
        int y_ = src.rows * 0.2;

        img(cv::Rect(x_, y_, src.cols, src.rows)) += src;

        HRect *rects;
        Shape *shapes;
        float *confs;
        int rsize = 0;

        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        rsize = detect(detector, img.data, img.cols, img.rows, img.step, &rects, &shapes, &confs);

        fprintf(fout, "%s\n%d\n", imageList[i].c_str(), rsize);

        for(int j = 0; j < rsize; j++){
            float minx =  FLT_MAX, miny =  FLT_MAX;
            float maxx = -FLT_MAX, maxy = -FLT_MAX;

            for(int p = 0; p < shapes[j].ptsSize; p++){
                HPoint2f pt = shapes[j].pts[p];

                minx = HU_MIN(minx, pt.x);
                maxx = HU_MAX(maxx, pt.x);
                miny = HU_MIN(miny, pt.y);
                maxy = HU_MAX(maxy, pt.y);
            }

            int faceSize = HU_MAX(maxx - minx + 1, maxy - miny + 1);
            int cx = (minx + maxx) * 0.5f;
            int cy = (miny + maxy) * 0.5f;

            int x0 = cx - (faceSize >> 1);
            int x1 = x0 + faceSize;
            int y0 = cy - 0.7f * faceSize;
            int y1 = cy + (faceSize >> 1);

            x0 = HU_MAX(x0, x_);
            x1 = HU_MIN(x1, src.cols + x_ - 1);
            y0 = HU_MAX(y0, y_);
            y1 = HU_MIN(y1, src.rows + y_ - 1);

            fprintf(fout, "%d %d %d %d %f\n", x0 - x_, y0 - y_, x1 - x0 + 1, y1 - y0 + 1, confs[j]);
        }

        if(rsize > 0){
            delete [] rects;
            delete [] shapes;
            delete [] confs;
        }
    }

    fclose(fout);

    release(&detector);

    return 0;
}


int main(int argc, char **argv){
#if defined(MAIN_TRAIN)
    main_train(argc, argv);

#elif defined(MAIN_DETECT_IMAGES)
    main_detect_images(argc, argv);

#elif defined(MAIN_DETECT_VIDEO)
    main_detect_video(argc, argv);

#elif defined(MAIN_GENERATE_SAMPLES)
    main_generate_samples(argc, argv);

#elif defined(MAIN_CALC_RATE)
    main_detect_images_fddb(argc, argv);

#endif

    return 0;
}



void extract_face_from_image(cv::Mat &src, cv::Rect &rect, cv::Mat &patch, Shape &shape){
    int x0 = rect.x;
    int y0 = rect.y;
    int x1 = rect.x + rect.width  - 1;
    int y1 = rect.y + rect.height - 1;

    int width  = src.cols;
    int height = src.rows;
    int w, h;

    int ptsSize;

    int bl = 0, bt = 0, br = 0, bb = 0;


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

    patch = cv::Mat(rect.height, rect.width, src.type(), cv::Scalar::all(0));

    w = rect.width - bl - br;
    h = rect.height - bt - bb;

    patch(cv::Rect(bl, bt, w, h)) += src(cv::Rect(x0, y0, w, h));

    ptsSize = shape.ptsSize;

    for(int i = 0; i < ptsSize; i++){
        shape.pts[i].x += (bl - x0);
        shape.pts[i].y += (bt - y0);
    }
}


static void transpose_wx8(const uint8_t* src, int sStride,
        uint8_t* dst, int dStride, int width)
{
    int i;

    for (i = 0; i < width; i++) {
        dst[0] = src[0 * sStride];
        dst[1] = src[1 * sStride];
        dst[2] = src[2 * sStride];
        dst[3] = src[3 * sStride];
        dst[4] = src[4 * sStride];
        dst[5] = src[5 * sStride];
        dst[6] = src[6 * sStride];
        dst[7] = src[7 * sStride];

        src++;
        dst += dStride;
    }
}



static void transpose_wxh(const uint8_t* src, int sStride,
        uint8_t* dst, int dStride,
        int width, int height)
{
    int i,j;

    for (i = 0; i < width; i++) {
        j = 0;
        for (; j <= height - 4; j += 4) {
            dst[i * dStride + j + 0] = src[(j + 0) * sStride + i];
            dst[i * dStride + j + 1] = src[(j + 1) * sStride + i];
            dst[i * dStride + j + 2] = src[(j + 2) * sStride + i];
            dst[i * dStride + j + 3] = src[(j + 3) * sStride + i];
        }

        for (; j < height; ++j) {
            dst[i * dStride + j] = src[j * sStride + i];
        }
    }
}


static void transpose_plane(const uint8_t* src, int sStride,
        uint8_t* dst, int dStride,
        int width, int height)
{
    int i = height;

    while (i >= 8) {
        transpose_wx8(src, sStride, dst, dStride, width);
        src += 8 * sStride;
        dst += 8;
        i -= 8;
    }

    if (i > 0) {
        transpose_wxh(src, sStride, dst, dStride, width, i);
    }
}



static void rotation_plane_90(const uint8_t* src, int sStride,
        uint8_t* dst, int dStride,
        int width, int height)
{
    src       += sStride * (height - 1);
    sStride = -sStride;
    transpose_plane(src, sStride, dst, dStride, width, height);
}


static void rotation_plane_270(const uint8_t* src, int sStride,
        uint8_t* dst, int dStride,
        int width, int height)
{
    dst       += dStride * (width - 1);
    dStride = -dStride;
    transpose_plane(src, sStride, dst, dStride, width, height);
}


static void mirror_copy_row(const uint8_t* src, uint8_t* dst, int width)
{
    int i;
    for (i = 0; i < width; i++){
        dst[i] = src[width - 1 - i];
    }
}


static void rotate_plane_180(const uint8_t* src, int sStride,
        uint8_t* dst, int dStride,
        int width, int height)
{
    uint8_t* row           = (uint8_t*)malloc(width);
    const uint8_t* src_bot = src + sStride * (height - 1);
    uint8_t* dst_bot       = dst + dStride * (height - 1);
    int half_height     = (height + 1) >> 1;
    int y;

    void(*ptr_mirror_copy_row)(const uint8_t* src, uint8_t* dst, int width) = mirror_copy_row;

    for (y = 0; y < half_height; ++y) {
        ptr_mirror_copy_row(src, row, width);
        src += sStride;
        ptr_mirror_copy_row(src_bot, dst, width);
        dst += dStride;

        memcpy(dst_bot, row, width);

        src_bot -= sStride;
        dst_bot -= dStride;
    }

    free(row);
}


static void copy_plane(uint8_t* src, int sStride, uint8_t* dst, int dStride, int width, int height)
{
    int y;
    for (y = 0; y < height; y++) {
        memcpy(dst, src, width);
        src += sStride;
        dst += dStride;
    }
}


void rotate_with_degree(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int &dstw, int &dsth, int &dsts, int degree){

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

        rotation_plane_90(src, srcs, dst, dsts, srcw, srch);
    }
    else if(degree == 2){
        dstw = srcw;
        dsth = srch;
        dsts = srcs;
        rotate_plane_180(src, srcs, dst, dsts, srcw, srch);
    }
    else if(degree == 3){
        dstw = srch;
        dsth = srcw;
        dsts = srch;

        rotation_plane_270(src, srcs, dst, dsts, srcw, srch);
    }
}


