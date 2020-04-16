#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv){
    if(argc != 3){     //注意这里的参数为3，是因为argv[0]是指向输入的程序路径及名称：./orb_cv
        cout << "usage: feature_extraction img1 img2 "<< endl;
        return 1;
    }
   //读取图像
   Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
   Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
   assert(img_1.data != nullptr && img_2.data != nullptr);
   //在程序运行时cv::assert()计算括号内的表达式，如果表达式为FALSE (或0), 程序将报告错误，并终止执行。
   //如果表达式不为0，则继续执行后面的语句。

   //初始化
   vector<KeyPoint> keypoints_1, keypoints_2;   //关键点/角点
   /**
   opencv中keypoint类的默认构造函数为：
   CV_WRAP KeyPoint() : pt(0,0), size(0), angle(-1), response(0), octave(0), class_id(-1) {}
   pt(x,y):关键点的点坐标； // size():该关键点邻域直径大小； // angle:角度，表示关键点的方向，值为[0,360)，负值表示不使用。
   response:响应强度，选择响应最强的关键点;   octacv:从哪一层金字塔得到的此关键点。
   class_id:当要对图片进行分类时，用class_id对每个关键点进行区分，默认为-1。
   **/
   Mat descriptors_1, descriptors_2;      //描述子
   //创建ORB对象，参数为默认值
   Ptr<FeatureDetector> detector = ORB::create();
   Ptr<DescriptorExtractor> descriptor = ORB::create();
   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
   /**
   “Ptr<FeatureDetector> detector = ”等价于 “FeatureDetector * detector =”
   Ptr是OpenCV中使用的智能指针模板类，可以轻松管理各种类型的指针。
   特征检测器FeatureDetetor是虚类，通过定义FeatureDetector的对象可以使用多种特征检测及匹配方法，通过create()函数调用。
   描述子提取器DescriptorExtractor是提取关键点的描述向量类抽象基类。
   描述子匹配器DescriptorMatcher用于特征匹配，"Brute-force-Hamming"表示使用汉明距离进行匹配。
   **/

   //第一步，检测Oriented Fast角点位置
   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   detector->detect(img_1, keypoints_1);     //对参数1图像进行特征的提取，并存放入参数2的数组中
   detector->detect(img_2, keypoints_2);

   //第二步，根据角点计算BREIF描述子
   descriptor->compute(img_1, keypoints_1, descriptors_1);   //computer()计算关键点的描述子向量（注意思考参数设置的合理性）
   descriptor->compute(img_2, keypoints_2, descriptors_2);
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;
   Mat outimg1;
   drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
   imshow("ORB features", outimg1);
   imwrite("feaure1.png", outimg1);

   //第三步， 对两幅图像中的描述子进行匹配，使用hamming距离
   vector<DMatch> matches;    //DMatch是匹配关键点描述子 类, matches用于存放匹配项
   t1 = chrono::steady_clock::now();
   matcher->match(descriptors_1, descriptors_2, matches); //对参数1 2的描述子进行匹配，并将匹配项存放于matches中
   t2 = chrono::steady_clock::now();
   time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "match the ORB cost: " << time_used.count() << "seconds. " << endl;

   //第四步，匹配点对筛选
   //计算最小距离和最大距离
   auto min_max = minmax_element(matches.begin(), matches.end(),
       [](const DMatch &m1, const DMatch &m2){ return m1.distance < m2.distance; });
   // auto 可以在声明变量的时候根据变量初始值的类型自动为此变量选择匹配的类型
   // minmax_element()返回指向范围内最小和最大元素的一对迭代器。参数1 2为起止迭代器范围
   // 参数3是二进制函数，该函数接受范围内的两个元素作为参数，并返回可转换为bool的值。
   // 返回的值指示作为第一个参数传递的元素是否小于第二个。该函数不得修改其任何参数。
   double min_dist = min_max.first->distance;  // min_max存储了一堆迭代器，first指向最小元素
   double max_dist = min_max.second->distance; // second指向最大元素

   printf("-- Max dist : %f \n", max_dist);
   printf("-- Min dist : %f \n", min_dist);

   //当描述子之间的距离大于两倍最小距离时，就认为匹配有误。但有时最小距离会非常小，所以要设置一个经验值30作为下限。
   vector<DMatch> good_matches;  //存放良好的匹配项
   for(int i = 0; i < descriptors_1.rows; ++i){
       if(matches[i].distance <= max(2 * min_dist, 30.0)){
           good_matches.push_back(matches[i]);
       }
   }

   //第五步，绘制匹配结果
   Mat img_match;         //存放所有匹配点
   Mat img_goodmatch;     //存放好的匹配点
   // drawMatches用于绘制两幅图像的匹配关键点。
   // 参数1是第一个源图像，参数2是其关键点数组；参数3是第二张原图像，参数4是其关键点数组
   // 参数5是两张图像的匹配关键点数组,参数6用于存放函数的绘制结果
   drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
   drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
   imshow("all matches", img_match);
   imshow("good matches", img_goodmatch);
   imwrite("match1.png", img_match);
   imwrite("goodmatch1.png", img_goodmatch);

   waitKey(0);

    return 0;
}