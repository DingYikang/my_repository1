#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

int main(int argc, char **argv){
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    double _sigma = 1.0;
    double inv_sigma = 1 / _sigma;
    int iteration = 100;
    int N = 100;
    vector<double> data_x ;
    vector<double> data_y ;
    cv::RNG rng;

    // 获得初始数据
    for(int i = 0; i < N; ++i){
        double xi = i/100.0;
        double yi = exp(ar*xi*xi + br*xi + cr) + rng.gaussian(_sigma * _sigma);
        data_x.push_back(xi);
        data_y.push_back(yi);
    }

    double cost, lastcost;
    // 开始迭代
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(size_t iter = 0; iter < iteration; ++iter){
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        cost = 0;
        double error = 0;

        for(size_t i = 0; i < N; ++i){
            Eigen::Vector3d J = Eigen::Vector3d::Zero();
            double ei = data_y[i] - exp(ae*data_x[i]*data_x[i] + be*data_x[i] + ce);

            J[0] = - data_x[i]*data_x[i] * exp(ae*data_x[i]*data_x[i] + be*data_x[i] + ce);
            J[1] = - data_x[i] * exp(ae*data_x[i]*data_x[i] + be*data_x[i] + ce);
            J[2] = - exp(ae*data_x[i]*data_x[i] + be*data_x[i] + ce);

            H += inv_sigma * inv_sigma * J * J.transpose();
            b += - inv_sigma * inv_sigma * ei *J;

            cost += ei * ei;
        }

        // 求解增量方程
        Eigen::Vector3d dx = H.ldlt().solve(b);

        // 对计算结果进行判断
        if(isnan(dx[0])){
            cout << "result is not a nummber. "<< endl;
            break;
        }

        if(iter > 0 && cost >= lastcost){
            cout << "cost : " << cost << "\t\t lastcost : "<< lastcost << "break." << endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];



        cout << "cost : "<< "\t\tlastcost : " << "\t\tae" << "\t\tbe"<<"\t\tce" << endl;
        cout << "  " << cost << "\t\t  "<< lastcost << "\t\t  " << ae << "\t\t  " << be
            << "\t\t  " << ce << endl;

        lastcost = cost;
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> timeused = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "use time : " << timeused.count() << endl;
    cout << "final abc = " << ae <<"\t"<< be <<"\t" << ce << endl;

    return 0;
}