#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>   // 具有格式化输出功能
#include <pangolin/pangolin.h>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;  // 存放像素点坐标的数组

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline   双目相机基线
double baseline = 0.573;
// paths
string left_file = "/home/yikang/cppSpace/slam/ch8/left.png";
string disparity_file = "/home/yikang/cppSpace/slam/ch8/disparity.png";
boost::format fmt_others("/home/yikang/cppSpace/slam/ch8/%06d.png");    // other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/// class for accumulator jacobians in parallel
// 定义求雅克比的类
class JacobianAccumulator {
public:
    // 构造函数
    JacobianAccumulator(
        const cv::Mat &img1_,             // 图像 1
        const cv::Mat &img2_,             // 图像 2
        const VecVector2d &px_ref_,       //  参考点像素坐标 数组
        const vector<double> depth_ref_,  // 参考点深度 数组
        Sophus::SE3d &T21_) :   // 坐标系1到坐标系2的变换矩阵
        img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }

    /// accumulate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

    /// get hessian matrix
    Matrix6d hessian() const { return H; }

    /// get bias
    Vector6d bias() const { return b; }

    /// get total cost
    double cost_func() const { return cost; }

    /// get projected points
    VecVector2d projected_points() const { return projection; }

    /// reset h, b, cost to zero
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;   // 标准互斥类型
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};

/**====
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);

/**==
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 **/
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);

// bilinear interpolation  双线性插值
// 双线性内插法利用待求像素四个相邻像素的灰度在两个方向上做线性内插，在光流法求取某像素位置的灰度值时同样用到了二维线性插值。
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check 边界检测
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)]; // data为指针，指向定位的像素位置
    // step()函数，返回像素行的实际宽度
    float xx = x - floor(x);   // floor()函数返回不大于x的最大整数
    float yy = y - floor(y);   // xx 和 yy 就是小数部分
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}


///////////  主函数
int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0);  // 读取灰度图像
    cv::Mat disparity_img = cv::imread(disparity_file, 0);  // 读取视差图像

    // randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;   // 点的数量
    int boarder = 20;     // 边界的像素数 ，在这里表示留空边上的一部分区域，不在边上取点
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        // rng.uniform()函数返回区间内均匀分布的随机数
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);   // 像素的视差
        double depth = fx * baseline / disparity; // 双目视觉中由视差到深度的计算
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }
    return 0;
}


///////  求解图像块的雅克比矩阵和增量方程
void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {
    // 求一个图像块内的雅克比矩阵的累积，为了解决单个像素在直接法中不具有代表性的缺点

    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; ++i) {

        // compute the projection in the second image
        Eigen::Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        // 计算参考点(第一幅图像)的三维坐标： ((x_i - c_x )/ fx), (y_i - c_y)/ fy, 1) * depth

        // 计算当前目标点(第二幅图像)的三维坐标
        Eigen::Vector3d point_cur = T21 * point_ref;

        if (point_cur[2] < 0)   // depth invalid
            continue;
        // 计算第i个参考点对应的目标点的像素坐标
        float u = fx * point_cur[0] / point_cur[2] + cx;  // u = c_x + fx * (x / z)
        float v = fy * point_cur[1] / point_cur[2] + cy;  // v = c_y + fy * (y / z)

        // 舍弃越界的像素点坐标
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
            v > img2.rows - half_patch_size)
            continue;

        // prijection 存放的是像素点坐标
        projection[i] = Eigen::Vector2d(u, v);
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
            Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;

        // 计数共有多少个良好的目标点
        cnt_good++;

        // compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {

                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                Matrix26d J_pixel_xi;   // 像素坐标对相机位姿李代数的一阶变化关系 : \frac{\partial u}{\partial \Delta epslon}
                Eigen::Vector2d J_img_pixel;  // 对应位置的像素梯度 ： \frac{\partial I}{\partial u}

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
                );

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();  // H 矩阵 ： 2*2
                bias += -error * J;            // b 矩阵 ： - e * J (有时也写作 b = - f * J )
                cost_tmp += error * error;     // 误差的二范数 (累加后是所有好的匹配点的误差二范数之和)
            }
    }

    if (cnt_good) {  // 如果好的目标点不为0，也就是J和b都有计算，那么进行以下操作
        // 计算最终的 H矩阵、b矩阵和误差二范数

        unique_lock<mutex> lck(hessian_mutex);
        // unique_lock 是为了避免 mutex 忘记释放锁。在对象创建时自动加锁，对象释放时自动解锁。
        // std::mutex类是一个同步原语，可用于保护共享数据被同时由多个线程访问。std::mutex提供独特的，非递归的所有权语义。
        // std::mutex是C++11中最基本的互斥量，std::mutex对象提供了独占所有权的特性，不支持递归地对std::mutex对象上锁。

        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good; // 本图像块的像素平均误差二范数
    }
}

//////////////// 单层直接法
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {

    const int iterations = 10;
    double cost = 0, lastCost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();
        // cv::parallel_for_是opencv封装的一个多线程接口，利用这个接口可以方便实现多线程，不用考虑底层细节
        // 下面这条语句相当于是此次迭代中的jacobian部分并行计算完了
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
                          // bind()函数起绑定效果，占位符std::placeholders::_1表示第一个参数对应jaco.accu::accumulate_jacobian的第一个参数
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // 求解增量方程
        Vector6d update = H.ldlt().solve(b);;
        T21 = Sophus::SE3d::exp(update) * T21;   // 更新位姿
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3) {
            // converge
            break;
        }

        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n" << T21.matrix() << endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // plot the projected pixels here
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i) {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey();
}


///////////////// 多层直接法
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {

    // 设置图像金字塔参数
    int pyramids = 4;   // 金字塔共有4层
    double pyramid_scale = 0.5;  // 每一层缩放比率是0.5
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // 创建图像金字塔
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            // 第一层，底层是原图像
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            // 上面的层使用resize()函数进行创建
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    // 由粗至精进行求解
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // 存放该层的目标点
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        // 不同的层上面由于进行了缩放，相机内参也相应的进行了改变
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        // 调用单层直接法进行求解
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }

}