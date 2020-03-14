#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.h"
using namespace std;
using namespace Eigen;

//演示sophus的基本用法
int main(int argc, char **argv)
{
    Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();    //绕Z轴旋转90°
    Quaterniond q(R);                                                           //四元数
    Sophus::SO3 SO3_R(R);                                                       //SO3直接从旋转矩阵构造
    Sophus::SO3 SO3_q(q);                                                       //可以通过四元数构造
    //展示两者是等价的
    cout << "SO(3) from matrix:\n " << SO3_R.matrix() << endl;
    cout << "SO(3) from quaternion:\n " << SO3_q.matrix() << endl;
    cout << " they are equal !" << endl;

    //使用对数映射得到李代数
    Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl;
    //hat是向量到反对称矩阵
    cout << "so3 hat= \n" << Sophus::SO3::hat(so3) << endl;
    //vee为反对称矩阵到向量
    cout << "so3 hat vee = " << Sophus::SO3::vee(Sophus::SO3::hat(so3)).transpose() << endl;

    Vector3d update_so3(1e-4, 0, 0);                           //假设更新向量（扰动向量）这么多
    Sophus::SO3 SO3_updated = Sophus::SO3::exp(update_so3) * SO3_R;
    cout << "SO3 updated = \n" << SO3_updated.matrix() << endl;

    cout << "*********************************************************" << endl;
    //对SE3操作一样的
    Vector3d t(1, 0, 0);
    Sophus::SE3 SE3_Rt(R, t);               //由R，t构造SE3
    Sophus::SE3 SE3_qt(q, t);               //由q,t构造SE3
    cout << "SE3 from R, t = " << SE3_Rt.matrix() << endl;
    cout << "SE3 from q, t = " << SE3_qt.matrix() << endl;
    //李代数时一个六维向量，方便起见先typedef一下
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    cout << "se3 hat = \n" << Sophus::SE3::hat(se3) << endl;
    cout << "se3 hat vee =\n" << Sophus::SE3::vee(Sophus::SE3::hat(se3)).transpose() << endl;

    //演示更新
    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0, 0) = 1e-4;
    Sophus::SE3 SE3_updated = Sophus::SE3::exp(update_se3) * SE3_Rt;
    cout << "SE3 update = " << endl << SE3_updated.matrix() << endl;

    return 0;
}