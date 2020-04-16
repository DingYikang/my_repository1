#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"  // 重投影误差模型类的头文件
#include <chrono>

using namespace std;

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;
        return 1;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    BALProblem bal_problem(argv[1]); // 从文件中读入数据，并生成bal_problem
    bal_problem.Normalize();         // 对原始数据进行归一化，将路标点中心置零，做一个合适尺度的放缩
    bal_problem.Perturb(0.1, 0.5, 0.5); // 设置噪声的信息 分别为位姿-旋转、位姿-平移、路标点噪声的sigma值
    bal_problem.WriteToPLYFile("initial.ply");   // 初始点云数据
    SolveBA(bal_problem);                        // 计算BA问题
    bal_problem.WriteToPLYFile("final.ply");     // 最终点云数据

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> timeuesd = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Ceses method costing time : \t" << timeuesd.count() << " s." << endl;

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    const double *observations = bal_problem.observations();
    ceres::Problem problem;

    ////// 构造BA问题
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        ceres::CostFunction *cost_function;

        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual 误差
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);

        // If enabled use Huber's loss function. 使用Huber核函数
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double *point = points + point_block_size * bal_problem.point_index()[i];

        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}