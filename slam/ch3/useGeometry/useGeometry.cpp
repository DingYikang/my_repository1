//本程序将演示Geometry几何模块的使用
#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>
//Geometry模块提供了各种旋转和平移的表示
using namespace Eigen;

int main(int argc, char ** argv)
{
    //3d旋转矩阵可以直接使用Matrix3d或者Matrix3f
    Matrix3d rotation_matrix = Matrix3d :: Identity();

    //旋转向量使用AngleAxis，运算可以当做矩阵
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0,0,1));     //眼Z轴旋转45°
    cout.precision(3);                                         //输出精度为小数点后两位
    cout << "rotation matrix = \n" << rotation_vector.matrix() << endl;
    //用matrix转换成矩阵可以直接赋值
    rotation_matrix = rotation_vector.toRotationMatrix();

    //使用Amgleanxis可以进行坐标变换
    Vector3d v(1, 0, 0);
    Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;

    //使用旋转矩阵
    v_rotated = rotation_matrix * v;
    cout << "(1,0,0) after rotation (by matrix) = " << v_rotated.transpose() << endl;

    //欧拉角：可以将矩阵直接转换成欧拉角
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);       //按照ZYX顺序
    cout << "yaw pitch row = "<< euler_angles.transpose() << endl;

    //欧式变换矩阵使用Eigen::Isometry
    Isometry3d T = Isometry3d::Identity();      //实质为4*4的矩阵
    T.rotate(rotation_vector);                  //按照rotation_vector进行转化
    T.pretranslate(Vector3d(1, 3, 4));          //平移向量设为（1， 3， 4）
    cout << "Transform matrix = \n" << T.matrix() <<endl;

    //变换矩阵进行坐标变换
    Vector3d v_transformed = T *v;
    cout << "v transormed =" << v_transformed.transpose() << endl;

    //四元数
    //直接把AngleAxis赋值给四元数，反之亦然
    Quaterniond q = Quaterniond(rotation_vector);
    cout << "quaternion from rotation vector = " << q.coeffs().transpose() << endl;
    q = Quaterniond(rotation_matrix);
    cout << "quaternion from rotation matrix = "<< q.coeffs().transpose() << endl;

    //使用四元数旋转一个向量，使用重载的乘法即可
    v_rotated = q * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
    cout << "should be equal to " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl;

    return 0;
}
