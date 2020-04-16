#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>  // Boost是为C++语言标准库提供扩展的一些C++程序库的总称。
//format.hpp实现类似printf的格式化对象，可以把参数格式化到一个字符串，而且是完全类型安全的。
#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>

using namespace std;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;  // 双精度浮点型三维特殊变换群
typedef Eigen::Matrix<double, 6, 1> Vector6d;   // 位姿向量

// 在pangolin中画图，已写好，无需调整
void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char **argv)
{
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图动态数组和深度图动态数组
    TrajectoryType poses;         // 相机位姿(类型是SE3D的动态数组)

    ifstream fin("/home/yikang/cppSpace/slam/ch5/RGBD/pose.txt");
    if (!fin)
    {
        cerr << "请在有pose.txt的目录下运行此程序" << endl;   //报错提醒
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt("/home/yikang/cppSpace/slam/ch5/RGBD/%s/%d.%s"); //fmt中的数据存放类型是字符串、整型、字符串。结合./和/和.正好是图像所在路径以及名称。
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        //format.str()直接将format中的内容转换成字符串，即成为读取图像的路径名称
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1)); // 使用-1读取原始图像

        double data[7] = {0};
        for (auto &d:data)
        //基于范围的for循环，表示从data数组的第一项开始循环遍历。auto表示自动根据后面的元素获得符合要求的类型
        //auto用来声明自动变量。它是存储类型标识符，表明变量(自动)具有本地范围
        //块范围的变量声明(如for循环体内的变量声明)默认为auto存储类型。
        //其实大多普通声明方式声明的变量都是auto变量,他们不需要明确指定auto关键字，默认就是auto。
        //auto变量在离开作用域时会被程序自动释放，不会发生内存溢出情况(除了包含指针的类)。
        //使用auto变量的优势是不需要考虑变量是否被释放，比较安全。
        //auto变量在函数结束时即释放了，再次调用这个函数时，又重新定义了一个新的变量
            fin >> d;
        //改变d变量，data变量也随之改变，因为d是data的引用
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),    //四元数,data[6]是实部
                          Eigen::Vector3d(data[0], data[1], data[2]));  //平移向量
        poses.push_back(pose);
    }

    // 计算点云并拼接
    // 相机内参
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000);    //reserve()表示容器预留空间，但不是真正的创建对象

    for (int i = 0; i < 5; i++) {
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Sophus::SE3d T = poses[i];
        //开始遍历像素
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                //ptr()函数访问任意一行像素的首地址，特别方便图像的一行一行的横向访问。
                //ptr()函数访问效率比较高，程序也比较安全，有越界判断。
                if (d == 0) continue; // 为0表示没有测量到
                //计算点云
                //像素坐标转换为世界坐标
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;   //毫米化为米
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                //转换为相机坐标
                Eigen::Vector3d pointWorld = T * point;
                // cout << "pointWord[u][v]: " << pointWorld << endl;

                Vector6d p;
                p.head<3>() = pointWorld;     //返回*this的前3个系数的固定大小的表达式。
                p[5] = color.data[v * color.step + u * color.channels()];   // blue
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
                //将彩色点云信息压入数组
                pointcloud.push_back(p);
            }
    }

    cout << "点云共有" << pointcloud.size() << "个点." << endl;
    showPointCloud(pointcloud);  //自动绘图
    return 0;
}
//可以拿来直接用的Pangolin画图程序
void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud)
{
    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}