#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>
#include <chrono>

#include "common.h"
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

/// 相机参数（姿态和内参）结构
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    /// 构造函数，从内存中读取数据并进行初始化
    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2])); // 1-3 组成的旋转向量变换为旋转矩阵
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);        // 4-6 组成的评议向量
        focal = data_addr[6];            // 一维的焦距
        k1 = data_addr[7];               // k1 , k2 径向畸变参数
        k2 = data_addr[8];
    }

    /// 将估计值放入内存函数
    void set_to(double *data_addr) {
        auto r = rotation.log();    // 对数映射，旋转矩阵变为旋转向量格式
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    SO3d rotation;   // 旋转矩阵
    Vector3d translation = Vector3d::Zero();    // 平移向量
    double focal = 0;                      // 一维焦距
    double k1 = 0, k2 = 0;                 // 径向畸变参数
};

//////////////////// 自己定义图优化中的结点  /////////////////////
//////////// 注意结点包含两种：相机位姿和特征点空间位置 /////////////

/// 位姿加相机内参的顶点，9维，前三维为so3，接下去为t, f, k1, k2，前4个是外参，后三个是内参

class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> { // 注意BaseVertex的两个参数，维度和点的数据类型
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;    // 字节对齐

    VertexPoseAndIntrinsics() {}    //  构造函数

    // 顶点重置函数，设置优化变量的初始值(有格式化的模板了)
    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics();     // _estimate是优化变量，也就是顶点代表的内容
    }

    // 顶点更新函数，计算出增量后，通过这个函数对估计值进行调整 (这个函数写的值得借鉴)
    virtual void oplusImpl(const double *update) override {
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6];               // 注意update的类型一般是数组类型，和_estimate不是一个类型的
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    /// 根据估计值投影一个点，从三维路标点获得像素坐标点
    Vector2d project(const Vector3d &point) {
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2];   // pc是归一化坐标系中的点
        double r2 = pc.squaredNorm();   //  pc的二范数计算函数
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        // 返回值是像素坐标系中的像素坐标，之所以没有出现相机光心是因为我们在导入数据前对数据进行了归一化操作，把中心置零了
        return Vector2d(_estimate.focal * distortion * pc[0],
                        _estimate.focal * distortion * pc[1]);
    }

    // 读盘和写盘虚函数声明一下即可
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

///// 代表优化路标点位置的结点，两个参数是代表结点维度是3、结点数据类型是三维向量
class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;   // 字节对齐

    VertexPoint() {}               // 构造函数

    // 顶点重置函数，设置顶点的初始值
    virtual void setToOriginImpl() override {
        _estimate = Vector3d(0, 0, 0);
    }

    // 顶点更新函数，用于更新优化变量，具体书写方式应当考虑结点数据类型和update的数据类型
    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update[0], update[1], update[2]);
    }

    // 读盘和写盘虚函数声明一下即可
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

////////////////////// 定义边 /////////////////////////////
////////////// 图优化中的边是代表误差/观测值 /////////////////

// 这里定义的边继承自二元边，四个参数的意义是：①测量值是二维的，②测量值类型是二维数组，③连接顶点类型，④连接顶点类型
class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;   // 字节对齐

    // 误差计算函数，用于计算优化问题中的e，即使用当前顶点值计算的测量值和真实测量值之间的误差
    virtual void computeError() override {
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];   // _vertices[] 存储顶点信息
        auto v1 = (VertexPoint *) _vertices[1];
        auto proj = v0->project(v1->estimate());  // 调用顶点1的project函数将根据顶点2估计得到的空间的变为像素点
        _error = proj - _measurement;   // measurment 存储观测值 , _error 存储误差e
    }

    // 同样的，读盘和写盘函数声明一下即可
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

};

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    // 关于BALProblem的内容去看 common.cpp 文件
    BALProblem bal_problem(argv[1]);   // 读取txt文件中的数据，并初始化一个BALProblem对象
    bal_problem.Normalize();           // 对数据进行归一化
    bal_problem.Perturb(0.1, 0.5, 0.5);   // 像数据中加入高斯噪声，参数代表sigma
    bal_problem.WriteToPLYFile("initial.ply");  // 把原始的点云数据存为initial.ply文件
    SolveBA(bal_problem);          // 求解该问题，g2o method
    bal_problem.WriteToPLYFile("final.ply");    // 把求解之后数据存为点云文件

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> timeused = chrono::duration_cast<chrono::duration<double>>(t2 -t1);
    cout << "G2O method costing time : \t" << timeused.count() << " s." << endl;

    return 0;
}

/////////// 编写使用g2o对该优化问题进行求解的函数，这里有使用g2o的搭建过程 /////////////

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();  // 路标点的维度
    const int camera_block_size = bal_problem.camera_block_size();  // 相机参数维度
    double *points = bal_problem.mutable_points();          // 路标点起始地址
    double *cameras = bal_problem.mutable_cameras();        // 相机参数起始地址

    // 参数9和3代表优化变量相机参数9维，误差项参数3维
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

    // 创建总求解器：Solver (使用LM法)
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    // 上一行实际上是使用线性求解器初始化块求解器，用来初始化总求解

    g2o::SparseOptimizer optimizer;    // 创建稀疏优化器
    optimizer.setAlgorithm(solver);    // 设置优化方法（即求解器）
    optimizer.setVerbose(true);        // 设置优化过程输出信息

    /// 构建g2o问题
    const double *observations = bal_problem.observations(); // 观测数据起始地址
    // 结点
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;   // 结点1：相机参数
    vector<VertexPoint *> vertex_points;                        // 结点2：路标点
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);                                        // 定义结点编号
        v->setEstimate(PoseAndIntrinsics(camera));          // 设置结点初始值
        optimizer.addVertex(v);                             // 向图中添加顶点
        vertex_pose_intrinsics.push_back(v);                // 把结点存入栈
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());                  // 定义结点编号
        v->setEstimate(Vector3d(point[0], point[1], point[2]));   // 设定结点初始值

        // g2o在BA中需要手动设置待边缘化的顶点，如同理论分析的一致，我们将路标点进行边缘化
        v->setMarginalized(true);
        optimizer.addVertex(v);            // 把结点添加到图中
        vertex_points.push_back(v);        // 把结点存入栈
    }

    // 边
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);  // 设置连接顶点信息1
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);            // 设置连接顶点信息2
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1]));  // 定义观测值
        edge->setInformation(Matrix2d::Identity());                             // 定义协方差矩阵的逆
        edge->setRobustKernel(new g2o::RobustKernelHuber());                // 设置鲁邦核函数为Huber核函数
        optimizer.addEdge(edge);                                /// 将边添加到图中
    }

    optimizer.initializeOptimization();   // 初始化优化器
    optimizer.optimize(40);    // 设置迭代次数

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }
}
