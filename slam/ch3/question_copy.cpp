/***********************************************************************************
题目：
已知旋转矩阵定义是沿着Z轴旋转45°。请按照该定义初始化旋转向量、旋转矩阵、四元数、欧拉角。请编程实现：
1、以上四种表达方式的相互转换关系并输出
2、假设平移向量为（1,2,3）,请输出旋转矩阵和该平移矩阵构成的欧式变换矩阵，并根据欧式变换矩阵提取旋转向量及平移向量

本程序学习目标：
1、学习eigen中刚体旋转的四种表达方式，熟悉他们之间的相互转换关系
2、熟悉旋转平移和欧式变换矩阵的相互转换关系
******************************************************************************/

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

using std::cout;
using std::endl;

int main(int argc, char **argv) {
  // 定义绕Z轴旋转45°的旋转向量
  Eigen::AngleAxisd rotationVector(M_PI / 4, Eigen::Vector3d(0, 0, 1));
  // 打印旋转向量
  cout << "AlgleAxis : "
       << rotationVector.angle() * rotationVector.axis().transpose() << endl;
  // 旋转向量 -> 旋转矩阵
  Eigen::Matrix3d rotationMatrix = rotationVector.toRotationMatrix();
  // 打印旋转矩阵
  cout << "Rotation Matrix :\n" << rotationMatrix << endl;
  // 旋转向量 -> 四元数  即四元数可以由旋转矩阵初始化，也可以由旋转向量初始化
  Eigen::Quaterniond quaternion_M(rotationMatrix);
  // 四元数必须归一化
  quaternion_M.normalize();
  // 打印四元数，顺序(x, y, z, w).coeffs()函数只是一种输出方式
  cout << "Quaternion(x, y, z, w) : " << quaternion_M.coeffs().transpose()
       << endl;
  // 旋转矩阵 -> 欧拉角，顺序（Z, Y, X)
  Eigen::Vector3d eulerAngle = rotationMatrix.eulerAngles(2, 1, 0);
  // 打印欧拉角
  cout << "EulerAngle : " << eulerAngle.transpose() << endl;

  // 定义平移量t
  Eigen::Vector3d t(1, 2, 3);
  // 打印平移向量
  cout << "t : " << t.transpose() << endl;
  // 欧式变换矩阵
  Eigen::Isometry3d T(rotationMatrix);
  //添加平移向量
  T.pretranslate(t);
  // 打印欧式变换矩阵
  cout << "Transform Matrix :\n" << T.matrix() << endl;

  // 欧式变换矩阵提取旋转矩阵
  Eigen::Matrix3d rotationMatrix_T = T.rotation();
  // Eigen::Matrix3d rotationMatrix_T = T.matrix().block(0, 0, 3, 3);
  cout << "Rotation Matrix frome Transform Matrix :\n"
       << rotationMatrix_T << endl;
  // 欧式变换矩阵提取旋转向量
  Eigen::Vector3d rotationVector_T = T.translation();
  // Eigen::Vector3d rotationVector_T = T.matrix().block(0, 3, 3, 1);
  cout << "t from Transform Matrix : " << rotationVector_T.transpose() << endl;

  return 0;
}