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

    train_aligner(argv[2], 4, &aligner);

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

        if(frame.rows > 720) cv::resize(frame, frame, cv::Size(720 * frame.cols / frame.rows, 720));

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        gettimeofday(&stv, &tz);
        rsize = process_track(gray.data, gray.cols, gray.rows, gray.step, FT_IMAGE_NV21, 0, shapes);
        gettimeofday(&etv, &tz);

        printf("%.2f ms\n", 1000.0f * (etv.tv_sec - stv.tv_sec) + 0.001f * (etv.tv_usec - stv.tv_usec));

#if 0
        for(int r = 0; r < rsize; r++){
            cv::Rect rect;

            float minx = FLT_MAX, maxx = -FLT_MAX;
            float miny = FLT_MAX, maxy = -FLT_MAX;
            float faceSize;
            float cx, cy;

            float *shape = shapes + r * PTS_SIZE2;

            for(int p = 0; p < PTS_SIZE; p ++){
                float x = shape[p * 2];
                float y = shape[p * 2 + 1];

                minx = HU_MIN(minx, x);
                maxx = HU_MAX(maxx, x);
                miny = HU_MIN(miny, y);
                maxy = HU_MAX(maxy, y);
            }

            faceSize = HU_MAX(maxx - minx + 1, maxy - miny + 1);

            cx = 0.5f * (minx + maxx);
            cy = 0.5f * (miny + maxy);

            rect.x = cx - faceSize * 0.5f;
            rect.y = cy - faceSize * 0.5f;
            rect.width = faceSize;
            rect.height = faceSize;

            cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
        }

        /*
        char outfile[256];
        sprintf(outfile, "res/%05d.jpg", frameIdx);
        cv::imwrite(outfile, frame);
        // */
        cv::imshow("frame", frame);
        cv::waitKey(10);

#else
        for(int r = 0; r < rsize; r++){
            float *pts = shapes + PTS_SIZE2 * r;
            for(int p = 0; p < PTS_SIZE; p++)
                cv::circle(frame, cv::Point2f(pts[p * 2], pts[p * 2 + 1]), 4, cv::Scalar(0, 255, 2), -1);
        }

        cv::imshow("frame", frame);
        cv::waitKey(1);
#endif
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

        if(rsize > 0)
            delete [] shapes;

        if(img.cols > 720)
            cv::resize(img, img, cv::Size(720, img.rows * 720 / img.cols));

        cv::imshow("img", img);
        cv::waitKey();
    }

    release(&manager);

    return 0;
}


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
//#pragma omp parallel for num_threads(omp_get_num_procs() - 1)
    for(int i = 0; i < size; i++){
        char rootDir[128], fileName[128], ext[30], filePath[256];
        const char *imgPath = imgList[i].c_str();

        cv::Mat img = cv::imread(imgPath, 1);
        cv::Mat gray;
        Shape *shapes;

        printf("%s\n", imgPath);
        if(img.empty()){
            printf("Can't open image %s\n", imgPath);
            continue;
        }

        if(img.cols < 96 || img.rows < 96)
            continue;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        int rsize = align_face(manager, gray.data, gray.cols, gray.rows, gray.step, &shapes);

        analysis_file_path(imgPath, rootDir, fileName, ext);

        for(int j = 0; j < rsize; j++){
            float *pts = shapes[j].pts;
            int ptsSize = shapes[j].ptsSize;
            HRect rect = get_shape_rect(shapes[j]);

            if(rect.width < 100) continue;

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
        }

        if(rsize > 0)
            delete [] shapes;

        /*
#pragma omp critical
        {
            finished ++;
            printf("%d\r", finished), fflush(stdout);
        }
        //*/
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
