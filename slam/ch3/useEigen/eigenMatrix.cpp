#include <iostream>
using namespace std;

#include <ctime>
//eigen核心部分
#include <Eigen/Core>
//稠密矩阵的代数运算（逆、特征值等）
#include <Eigen/Dense>
using namespace Eigen;

#define MATRIX_SEZE 50

/***************************
  本程序演示了Eigen基本类型的使用 P48
***************************/

int main(int argc, char **argv)
{
    //声明一个2*3的矩阵，eigen中所有的向量矩阵均为Eigen::Matrix，是一个模板类。
    //三个参数分别为数据类型、行数、列数
    Matrix<float, 2, 3> matrix_23;

    //声明一个三维向量，Vector_3d实际上是一个通过typedef定义的内置类型，本质上仍然是Eigen::Matrix模板类
    Vector3d v_3d;
    //Matrix<float, 2, 3> vd_3d;

    //声明一个3*3的矩阵，并初始化为0矩阵
    Matrix3d matrix_33 = Matrix3d::Zero();

    //不确定矩阵大小时，可以使用动态大小的矩阵
    Matrix<double, Dynamic, Dynamic> matrix_dynamic;

    //或者更简单的方式
    MatrixXd matrix_x;

    //准备对Eigen矩阵初始化
    //输入数据
    matrix_23 << 1, 2, 3, 4, 5, 6;
    //shuchu
    cout << "matrix 2*3 from 1 to 6 : \n"<< matrix_23 << endl;

    //使用（）访问矩阵中的元素
    cout << "print matirx 2*3 : "<< endl;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            cout << matrix_23(i,j) << "\t";
        }
        cout << endl;
    }

    //初始化两个三维向量
    v_3d << 3, 2, 1;
    //vd_3d << 4, 5, 6;
    //不能混乘两种类型的矩阵，因此需要显示转换类型：.cast<type>
    Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << "[1,2,3;4,5,6]*[3,2,1] = "<< result.transpose() << endl;
    //.transpose()是转置函数
    cout << "not transpose [1,2,3;4,5,6]*[3,2,1] = "<< result << endl;

    Matrix<float, 2, 1> result2 = matrix_23 * v_3d.cast<float>();
    cout << "[1,2,3;4,5,6]*[3,2,1] = "<< result2.transpose() << endl;

    //假如我们搞错了矩阵的维度
    //Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

    /************
     下面演示一些矩阵运算
     *****************/
    //生成一个随机矩阵
    matrix_33 = Matrix3d::Random();
    cout << "random matrix: \n"<< matrix_33 << endl;
    cout << "transpose : \n"<< matrix_33.transpose() << endl;  //转置
    cout << "sum :" << matrix_33.sum() << endl;                //求和
    cout << "trace : "<< matrix_33.trace() << endl;            //求迹
    cout << "time 10: \n"<< 10 * matrix_33 << endl;            //数乘
    cout << "inverse : \n"<< matrix_33.inverse() << endl;      //求逆
    cout << "det : \n"<< matrix_33.determinant() << endl;      //求行列式

    //求特征值
    //实对称矩阵可以保证对角化成功
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);   //SelfAdjointEigenSolver<Matrix3d>y用于求解特征值和特征向量
    cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

    //练习求解方程
    //求解Matrix_NN * x = v_NN方程
    Matrix<double, MATRIX_SEZE, MATRIX_SEZE> matrix_NN
        = MatrixXd::Random(MATRIX_SEZE,MATRIX_SEZE);     //MATRIX_SEZE在前面宏定义中
    matrix_NN = matrix_NN *matrix_NN.transpose();        //此处理可以保证矩阵是半正定的
    Matrix<double, MATRIX_SEZE, 1> v_Nd = MatrixXd::Random(MATRIX_SEZE, 1);

    clock_t time_stt = clock();                          //用于计时

    //直接求逆
    Matrix<double, MATRIX_SEZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time of normal inverse is"
        << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = "<< x.transpose() << endl;

    //再用矩阵分解，例如QR分解求解，速度将会快很多
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time of Qr decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    //对于正定矩阵，还可以用cholesky分解来分解方程
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of ldlt decomposition is "
       << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    return 0;
}