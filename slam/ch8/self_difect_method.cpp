#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace std;
using namespace cv;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// cameral intinsics
double fx = 718.856, fy = 718.865, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// path
string file_1 = "/home/yikang/cppSpace/slam/ch8/left.png";
string file_2 = "/home/yikang/cppSpace/slam/ch8/right.png";
boost::format fmt_others("/home/yikang/cppSpace/slam/ch8/%06d.png");

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

//定义并行计算计算雅克比矩阵的类
class JacbianAcumulator{
    public:
    JacbianAcumulator(
        const cv::Mat & img1_,
        const cv::Mat & img2_,
        const VecVector2d & px_ref_,
        const vector<double> depth_ref_,
        Sophus::SE3d & T21_ ) :
        img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_){
            projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
        }

    // 计算一个区域内的雅克比
    void accumulate_jacobian(const cv::Range &range);

    // 获取海塞矩阵
    Matrix6d Hesaian() const{ return H; };

    /// get bias
    Vector6d bias() const { return b; }

    /// get total cost
    double cost_func() const { return cost; }

    /// get projected points
    VecVector2d projected_points() const { return projection; }

    void reset(){
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }


    private:
        const cv::Mat img1;
        const cv::Mat img2;
        const VecVector2d px_ref;
        const vector<double> depth_ref;
        Sophus::SE3d T21;
        VecVector2d projection;

        std::mutex hessian_mutex;
        Matrix6d H = Matrix6d::Zero();
        Vector6d b = Vector6d::Zero();
        double cost = 0;
};

void DirectPoseEstamiteMultiLayer(
    const Mat & imag1,
    const Mat & iamg2,
    const VecVector2d & px_ref,
    const vector<double> & depth_ref,
    Sophus::SE3d & T21
);

void DirectPoseEstamiteSingleLayer(
    const Mat & img1,
    const Mat & img2,
    const VecVector2d & px_ref,
    const vector<double> & depth_ref,
    Sophus::SE3d & T21
);

inline float GetPixelValue( const cv::Mat & img, float x, float y){
    //判断非法输入
    if(x < 0){
        x = 0;
    }
    if(y < 0){
        y = 0;
    }
    if(x > img.cols){
        x = img.cols - 1;
    }
    if(y > img.rows){
        y = img.rows - 1;
    }
    uchar * data = &img.data[int(x) * img.step + int(y)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0]+
        xx * (1-yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
};

int main(int argc, char * argv){

    cv::Mat left_img = imread(file_1, 0);
    cv::Mat disparity_img = imread(file_2, 0);
    assert(left_img.data != nullptr && disparity_img.data != nullptr);
    int border = 20;

    int npoints = 200;
    cv::RNG rng;
    VecVector2d px_ref;
    vector<double> depth_ref;
    // 利用随机数取200个点
    for(size_t i = 0; i < npoints; ++i){
        int x = rng.uniform(double(border), double(left_img.cols - border - 1));
        int y = rng.uniform(double(border), double(left_img.rows - border -1));
        double disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity;
        depth_ref.push_back(depth);
        px_ref.push_back(Eigen::Vector2d(x, y));
    }

    // 估计5幅图像的位姿
    Sophus::SE3d T_cur_ref;

    for(size_t i = 0; i < 6; ++i){
        cv::Mat img = imread((fmt_others % i).str(), 0);
        DirectPoseEstamiteSingleLayer(left_img, img, px_ref, depth_ref, T_cur_ref);
        DirectPoseEstamiteMultiLayer(left_img, img, px_ref, depth_ref, T_cur_ref);
    }

    return 0;
}


void DirectPoseEstamiteSingleLayer(
    const Mat & img1,
    const Mat & img2,
    const VecVector2d & px_ref,
    const vector<double> & depth_ref,
    Sophus::SE3d & T21){

    const int iterations = 10;
    double cost = 0, lastcost = 0;
    auto t1 = chrono::steady_clock::now();

    JacbianAcumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for(size_t iter = 0; iter < iterations; ++iter){
        jaco_accu.reset();

        cv::parallel_for_(cv::Range(0, px_ref.size()),
            std::bind(&JacbianAcumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.Hesaian();
        Vector6d b = jaco_accu.bias();

        // solve
        Vector6d update = H.ldlt().solve(b);
        // 更新位姿
        T21 *= Sophus::SE3d::exp(update);

        cost = jaco_accu.cost_func();

         if (std::isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastcost) {
            cout << "cost increased: " << cost << ", " << lastcost << endl;
            break;
        }
        if (update.norm() < 1e-3) {
            // converge
            break;
        }

        lastcost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n" << T21.matrix() << endl;
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> timeused = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "time for direct method single layer: " << timeused.count() << endl;

    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for(size_t i = 0; i < px_ref.size(); ++i){
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if(p_cur[0] > 0 && p_cur[1] > 0)
        {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]), cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("current", img2_show);
    waitKey();
}