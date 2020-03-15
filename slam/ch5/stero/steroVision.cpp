#include <opencv2/opencv.hpp>
//<opencv2/opencv.hpp>头文件包含了各个模块的头文件，原则上无论创建哪个模块的
//应用程序，仅写一句#include <opencv2/opencv.hpp>即可，以达到精简简化代码的效果
#include <vector>               //STL中的可变长动态数组头文件
#include <string>
#include <Eigen/Core>           //矩阵计算函数库Eigen的核心模块
#include <pangolin/pangolin.h>  //基于openGL的画图工具Pangolin头文件
#include <unistd.h>             // C++中提供对操作系统访问功能的头文件，如fork/pipe/各种I/O（read/write/close等等）

using namespace std;
using namespace Eigen;

// 记录准确文件路径
string left_file = "/home/yikang/cppSpace/slam/ch5/stero/left.png";
string right_file = "/home/yikang/cppSpace/slam/ch5/stero/right.png";

// 在pangolin中画图，已写好，无需调整
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main(int argc, char **argv)
{
    // 相机内参，一般为已知数据
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 双目相机基线，一般已知
    double b = 0.573;

    // 读取图像
    cv::Mat left = cv::imread(left_file, 0);   //imread()参数2为0时，表示返回灰度图像，默认值为1，代表返回彩色图像
    cv::Mat right = cv::imread(right_file, 0); //从文件路径中读取两幅图像，返回灰度图像
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 调用OpenCv中的SGBM算法，用于计算左右图像的视差
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);   //将视差的计算结果放入disparity_sgbm矩阵中
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f); //将矩阵disparity_sgbm转换为括号中的格式(32位空间的单精度浮点型矩阵)

    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud; //声明一个4维的双精度浮点型可变长动态数组

    // 如果自己的机器慢，可以把++v和++u改成v+=2, u+=2
    for (int v = 0; v < left.rows; ++v)
        for (int u = 0; u < left.cols; ++u) {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;
            //Mat.at<存储类型名称>(行，列)[通道]，用以遍历像素。省略通道部分时，可以看做二维数组简单遍历，例如M.at<uchar>(512-1,512*3-1)；

            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色。第四维数值归一化。

            // 根据双目模型计算 point 的位置
            double x = (u - cx) / fx;      //像素坐标转换为归一化坐标
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));  //计算各像素点深度
            //计算带深度信息的各点坐标
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            pointcloud.push_back(point);   //将各点信息压入点云数组
        }

    cv::imshow("disparity", disparity / 96.0); //输出显示disparuty，显示窗口命名为引号中的内容
    cv::waitKey(0);           //等待关闭显示窗口，括号内参数为零则表示等待输入一个按键才会关闭，为数值则表示等待X毫秒后关闭
    // 画出点云
    showPointCloud(pointcloud);
    return 0;
}
    //定义画出点云的函数,函数参数为四维动态数组的引用
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);   //创建一个Pangolin的画图窗口,声明命名以及显示的分辨率
    glEnable(GL_DEPTH_TEST);    //启用深度缓存。
    glEnable(GL_BLEND);         //启用gl_blend混合。Blend混合是将源色和目标色以某种方式混合生成特效的技术。
    //混合常用来绘制透明或半透明的物体。在混合中起关键作用的α值实际上是将源色和目标色按给定比率进行混合，以达到不同程度的透明。
    //α值为0则完全透明，α值为1则完全不透明。混合操作只能在RGBA模式下进行，颜色索引模式下无法指定α值。
    //物体的绘制顺序会影响到OpenGL的混合处理。
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  //混合函数。参数1是源混合因子，参数2时目标混合因子。本命令选择了最常使用的参数。

    //定义投影和初始模型视图矩阵
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        //对应为gluLookAt,摄像机位置,参考点位置,up vector(上向量)
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );
    //管理OpenGl视口的位置和大小
    pangolin::View &d_cam = pangolin::CreateDisplay()
        //使用混合分数/像素坐标（OpenGl视图坐标）设置视图的边界
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        //指定用于接受键盘或鼠标输入的处理程序
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        //清除屏幕
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //激活要渲染到视图
        d_cam.Activate(s_cam);
        //glClearColor：red、green、blue、alpha分别是红、绿、蓝、不透明度，值域均为[0,1]。
        //即设置颜色，为后面的glClear做准备，默认值为（0,0,0,0）。切记：此函数仅仅设定颜色，并不执行清除工作。
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        //glPointSize 函数指定栅格化点的直径。一定要在要在glBegin前,或者在画东西之前。
        glPointSize(2);
        //glBegin()要和glEnd()组合使用。其参数表示创建图元的类型，GL_POINTS表示把每个顶点作为一个点进行处理
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);  //在OpenGl中设置颜色
            glVertex3d(p[0], p[1], p[2]); //设置顶点坐标
        }
        glEnd();
        pangolin::FinishFrame();    //结束
        usleep(5000);   // sleep 5 ms
    }
    return;
}