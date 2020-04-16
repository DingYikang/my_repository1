#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv){
    if( argc != 2 ){
        cout << "usage: q2 test1.jpg" << endl;
        return 1;
    }

    Mat img_test = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    assert(img_test.data && "cannot load image!");

    //SIFT特征点提取
    Ptr<xfeatures2d::SIFT> siftdetector = cv::xfeatures2d::SIFT::create();
    vector<KeyPoint> Keypoint_sift;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    siftdetector->detect(img_test, Keypoint_sift);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> timeuesd = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Using SIFT algorithm cost time: " << timeuesd.count() << " seconds ." << endl;

    Mat siftkeypointMat;
    cv::drawKeypoints(img_test, Keypoint_sift, siftkeypointMat, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imshow("SFIT", siftkeypointMat);

    //ORB特征点提取
    Ptr<FeatureDetector> detector = ORB::create();
    vector<KeyPoint> Keypoint_orb;
    t1 = chrono::steady_clock::now();
    detector->detect(img_test, Keypoint_orb);
    t2 = chrono::steady_clock::now();
    timeuesd = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Using ORB algorithm cost time: " << timeuesd.count() << " seconds ." << endl;

    Mat orbkeypointMat;
    cv::drawKeypoints(img_test, Keypoint_orb, orbkeypointMat, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imshow("ORB", orbkeypointMat);
    waitKey(0);

    return 0;
}