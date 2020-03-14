#include <iostream>
#include <chrono>
#include <cstring>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char ** argv)
{
    cv::Mat image;
    string path = "/home/yikang/cppSpace/slam/ch5/imageBasics/ubuntu.png";
    image = imread(path);

    if(image.data == NULL)     //如果出了问题，很可能在NULL这里
    {
        cerr << "文件" << path << "不存在" << endl;
        return 0;
    }

    //文件读取顺利，需要输出一些基本信息
    cout << "宽度" << image.cols << ", 高度" << image.rows
        << "， 通道数 " << image.channels() << endl;
    //显示图像
    cv::imshow("image",image);
    cv::waitKey(0);

    //判断特殊情况
    if(image.type() != CV_8UC1 && image.type() != CV_8UC3)
    {
        cout << "请输入一张彩色图或者一张灰度图" << endl;
        return 0;
    }

    //遍历图像，访问图像像素
    //使用chrono计时
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(size_t y = 0; y < image.rows ; ++y)
    {
        unsigned char *row_ptr = image.ptr<unsigned char>(y);
        for(size_t x = 0; x < image.cols; ++x)
        {
            unsigned char *data_ptr = &row_ptr[x * image.channels()];

            for(int c=0; c != image.channels(); ++c)
            {
                unsigned char data = data_ptr[c];
            }
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast< chrono::duration <double> >(t2 - t1);
    cout << "遍历图像用时 " << time_used.count() << " 秒 " << endl;

    //关乎Mat的浅拷贝（只有文件头和指针被拷贝）
    cv::Mat image_another = image;
    //修改image.another会导致image发生变化
    image_another(cv::Rect(0, 0, 100, 100)).setTo(0);
    cv::imshow("image", image);
    cv::waitKey(0);

    //使用clone函数拷贝数据
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
    cv::imshow("image", image);
    cv::imshow("image_clone", image_clone);
    waitKey(0);

    cv::destroyAllWindows();
    return 0;
}