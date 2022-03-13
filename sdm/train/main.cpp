#include "face_manager.h"
#include <sys/time.h>

#define ALIGN_MODEL "align_model.dat"
#define TRACK_MODEL "track_model.dat"
#define DETECT_MODEL "detect_model.dat"


int main_train(int argc, char **argv){
    if(argc < 3){
        printf("Usage: %s  [flag] [image list]\n", argv[0]);
        return 1;
    }

    Aligner *aligner;
    int flag;

    flag = atoi(argv[1]);

    train(argv[2], flag, 4, &aligner);

    if(flag == 1)
        save(ALIGN_MODEL, aligner);
    else
        save(TRACK_MODEL, aligner);

    release(&aligner);

    return 0;
}


int main_detect_video2(int argc, char **argv){
    if(argc < 2){
        printf("Usage: %s [video]\n", argv[0]);
        return 1;
    }

    cv::VideoCapture cap;

    if(load_models(DETECT_MODEL, ALIGN_MODEL, TRACK_MODEL) != 0){
        return 2;
    }

    cap.open(argv[1]);
    if(! cap.isOpened()){
        printf("Can't open video %s\n", argv[1]);
        return 3;
    }

    int totalFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
    struct timeval  stv, etv;
    struct timezone tz;

    for(int frameIdx = 0; frameIdx < totalFrame; frameIdx++){
        cv::Mat frame, gray;
        float shapes[MAX_FACE_SIZE * PTS_SIZE2];
        int rsize;

        cap >> frame;

        if(frame.empty()) continue;

        /*
        cv::imshow("frame", frame);
        cv::waitKey();
        //*/

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        gettimeofday(&stv, &tz);
        rsize = process_track(gray.data, gray.cols, gray.rows, gray.step, FT_IMAGE_NV21, 0, shapes);
        gettimeofday(&etv, &tz);

        printf("%.2f ms\n", 1000.0f * (etv.tv_sec - stv.tv_sec) + 0.001f * (etv.tv_usec - stv.tv_usec));

        for(int r = 0; r < rsize; r++){
            float *pts = shapes + PTS_SIZE2 * r;
            for(int p = 0; p < PTS_SIZE; p++)
                cv::circle(frame, cv::Point2f(pts[p * 2], pts[p * 2 + 1]), 4, cv::Scalar(0, 255, 2), -1);
        }

        cv::imshow("frame", frame);
        cv::waitKey(5);
    }

    release_models();
    return 0;
}


int main_detect_video(int argc, char **argv){
    if(argc < 2){
        printf("Usage: %s [video]\n", argv[0]);
        return 1;
    }

    FaceManager *manager = NULL;
    cv::VideoCapture cap;

    if(load(DETECT_MODEL, NULL, TRACK_MODEL, &manager) != 0){
        return 2;
    }

    cap.open(argv[1]);
    if(! cap.isOpened()){
        printf("Can't open video %s\n", argv[1]);
        return 3;
    }

    int totalFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
    struct timeval  stv, etv;
    struct timezone tz;

    for(int frameIdx = 0; frameIdx < totalFrame; frameIdx++){
        cv::Mat frame, gray;
        Shape shapes[MAX_FACE_SIZE];
        int rsize;

        cap >> frame;

        if(frame.empty()) continue;

        /*
        cv::imshow("frame", frame);
        cv::waitKey();
        //*/

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        gettimeofday(&stv, &tz);
        rsize = track_mul_face(manager, gray.data, gray.cols, gray.rows, gray.step, shapes);
        gettimeofday(&etv, &tz);

        printf("%.2f ms\n", 1000.0f * (etv.tv_sec - stv.tv_sec) + 0.001f * (etv.tv_usec - stv.tv_usec));

        for(int r = 0; r < rsize; r++){
            Shape shape = shapes[r];
            for(int p = 0; p < shape.ptsSize; p++)
                cv::circle(frame, cv::Point2f(shape.pts[p], shape.pts[p + shape.ptsSize]), 4, cv::Scalar(0, 255, 2), -1);
        }

        cv::imshow("frame", frame);
        cv::waitKey();
    }

    release(&manager);
    return 0;
}


int main_detect_images(int argc, char **argv){
    if(argc < 3){
        printf("Usage: %s [image list] [out dir]\n", argv[0]);
        return 1;
    }

    std::vector<std::string> imgList;
    FaceManager *manager = NULL;
    int ret, size;

    if(load(DETECT_MODEL, ALIGN_MODEL, NULL, &manager) != 0){
        return 2;
    }

    ret = read_file_list(argv[1], imgList);

    size = imgList.size();

    for(int i = 0; i < size; i++){
        const char *imgPath = imgList[i].c_str();
        cv::Mat img = cv::imread(imgPath, 1);
        cv::Mat gray;
        Shape *shapes;

        if(img.empty()){
            printf("Can't open image %s\n", imgPath);
            continue;
        }

        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        int rsize = align_face(manager, gray.data, gray.cols, gray.rows, gray.step, &shapes);

        for(int j = 0; j < rsize; j++){
            float *pts = shapes[j].pts;
            int ptsSize = shapes[j].ptsSize;
            int radius = 3 * HU_MAX(img.cols / 720, 1) ;
            for(int p = 0; p < ptsSize; p++)
                cv::circle(img, cv::Point2f(pts[p], pts[p + ptsSize]), radius, cv::Scalar(0,  255, 0), -1);
        }

        if(img.cols > 720)
            cv::resize(img, img, cv::Size(720, img.rows * 720 / img.cols));

        cv::imshow("img", img);
        cv::waitKey();
    }

    release(&manager);

    return 0;
}


void normalize_sample(cv::Mat &src, cv::Mat &patch, int winSize, float factor, Shape &shape);
int main_gen_samples(int argc, char **argv){
    if(argc < 3){
        printf("Usage: %s [image list] [out dir]\n", argv[0]);
        return 1;
    }

    std::vector<std::string> imgList;
    FaceManager *manager = NULL;
    int ret, size;

    if(load(DETECT_MODEL, ALIGN_MODEL, NULL, &manager) != 0){
        return 2;
    }

    ret = read_file_list(argv[1], imgList);

    size = imgList.size();

    int finished = 0;
    const char *PREFIX = "mega";
    int index = 103126;

//#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < size; i++){
        char rootDir[128], fileName[128], ext[30], filePath[256];
        const char *imgPath = imgList[i].c_str();

        cv::Mat img = cv::imread(imgPath, 1);
        cv::Mat gray;
        Shape *shapes;

        if(img.empty()){
            printf("Can't open image %s\n", imgPath);
            continue;
        }

        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        int rsize = align_face(manager, gray.data, gray.cols, gray.rows, gray.step, &shapes);

        analysis_file_path(imgPath, rootDir, fileName, ext);

        for(int j = 0; j < rsize; j++){
            float *pts = shapes[j].pts;
            int ptsSize = shapes[j].ptsSize;
            HRect rect = get_shape_rect(shapes[j]);

            if(rect.width < 100) continue;
#if 1
            cv::Mat patch;

            normalize_sample(img, patch, 0, 3.0, shapes[j]);

            sprintf(filePath, "%s/%s_%06d.jpg", argv[2], PREFIX, index);
            ret = cv::imwrite(filePath, patch); assert(ret == 1);

            sprintf(filePath, "%s/%s_%06d.pts", argv[2], PREFIX, index);
            ret = write_pts_file(filePath, pts, ptsSize); assert(ret == 0);

            index ++;

#else
            if(j == 0){
                sprintf(filePath, "%s/%s.jpg", argv[2], fileName);
                ret = cv::imwrite(filePath, img); assert(ret == 1);

                sprintf(filePath, "%s/%s.pts", argv[2], fileName);
                ret = write_pts_file(filePath, pts, ptsSize); assert(ret == 0);
            }
            else {
                sprintf(filePath, "%s/%s_%d.jpg", argv[2], fileName, j);
                ret = cv::imwrite(filePath, img); assert(ret == 1);

                sprintf(filePath, "%s/%s_%d.pts", argv[2], fileName, j);
                ret = write_pts_file(filePath, pts, ptsSize); assert(ret == 0);
            }
#endif
        }

        if(rsize > 0)
            delete [] shapes;

#pragma omp critical
        {
            finished ++;
            printf("%d\r", finished), fflush(stdout);
        }
    }

    release(&manager);

    return 0;
}


int main(int argc, char **argv){
#if defined(MAIN_TRAIN)
    main_train(argc, argv);

#elif defined(MAIN_DETECT_VIDEO)
    main_detect_video2(argc, argv);

#elif defined(MAIN_DETECT_IMAGE)
    main_detect_images(argc, argv);

#elif defined(MAIN_GEN_SAMPLES)
    main_gen_samples(argc, argv);

#endif

    return 0;
}


void normalize_sample(cv::Mat &src, cv::Mat &patch, int winSize, float factor, Shape &shape){
    HRect rect;
    int width  = src.cols;
    int height = src.rows;
    int faceSize;

    int bl, bt, br, bb;
    int x0, y0, x1, y1;
    int w, h;

    float scale = 1;
    int ptsSize;

    rect = get_shape_rect(shape);

    faceSize = rect.width * factor;

    int cx = rect.x + (rect.width  >> 1);
    int cy = rect.y + (rect.height >> 1);

    x0 = cx - (faceSize >> 1);
    y0 = cy - (faceSize >> 1);
    x1 = x0 + faceSize - 1;
    y1 = y0 + faceSize - 1;

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

    cv::Rect rect1(bl, bt, w, h);
    cv::Rect rect2(x0, y0, w, h);

    patch = cv::Mat(faceSize, faceSize, src.type(), cv::Scalar::all(0));

    patch(rect1) = patch(rect1) + src(rect2);

    scale = 1;
    if(winSize != 0){
        scale = float(winSize) / faceSize;
        resize(patch, patch, cv::Size(scale * patch.cols, scale * patch.rows));
    }
    else if(faceSize > 2000){
        winSize = 2000;
        scale = float(winSize) / faceSize;
        resize(patch, patch, cv::Size(scale * patch.cols, scale * patch.rows));
    }
    else if(faceSize < 300){
        winSize = 300;
        scale = float(winSize) / faceSize;
        resize(patch, patch, cv::Size(scale * patch.cols, scale * patch.rows));
    }

    ptsSize = shape.ptsSize;

    for(int i = 0; i < ptsSize; i++){
        int xi = i << 1;
        int yi = xi + 1;
        shape.pts[xi] = (shape.pts[xi] - x0 + bl) * scale;
        shape.pts[yi] = (shape.pts[yi] - y0 + bt) * scale;
    }
}


