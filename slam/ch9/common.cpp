#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "common.h"
#include "rotation.h"
#include "random.h"

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value); // fscanf函数根据数据格式format，从fptr中读取数据到value中，遇到空格和换行结束
    if (num_scanned != 1)    // 如果读取未成功，则报错
        std::cerr << "Invalid UW data file. ";
}

// 为三维点数据加入噪声
void PerturbPoint3(const double sigma, double *point) {
    for (int i = 0; i < 3; ++i)
        point[i] += RandNormal() * sigma;
}

// 找到数组的中值
double Median(std::vector<double> *data) {
    int n = data->size();
    std::vector<double>::iterator mid_point = data->begin() + n / 2;  // 指向中间的迭代器
    std::nth_element(data->begin(), mid_point, data->end());
    // std::nth_element()函数用于排序第nth个元素（从0开始的索引），排序后__nth位置就是第nth大的元素，前面小，后面大，但是前后两区间内的大小未排序
    return *mid_point;
}

//// 定义BALProblem的构造函数，BALProblem类的声明在“commom.h”头文件中
BALProblem::BALProblem(const std::string &filename, bool use_quaternions) {
    FILE *fptr = fopen(filename.c_str(), "r");

    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filename;
        return;
    };

    // 从fptr中读取数据（计数），若读取失败则报错并终止程序
    // txt数据文件的 开头时三类数据的数量
    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    std::cout << "Header: " << num_cameras_
              << " " << num_points_
              << " " << num_observations_ << "\n" ;

    // 根据读取数据的数量，为路标点、相机位姿、观测数据开辟数组空间
    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    // 参数的总数考虑了数据的维度
    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    // 将观测数据读入前面开辟好的数组中
    // 数据一共有num_observatios_行，每一行数据依次是相机参数、路标点、观测值
    for (int i = 0; i < num_observations_; ++i) {
        FscanfOrDie(fptr, "%d", camera_index_ + i);
        FscanfOrDie(fptr, "%d", point_index_ + i);
        for (int j = 0; j < 2; ++j) {
            FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
        }
    }

    for (int i = 0; i < num_parameters_; ++i) {
        FscanfOrDie(fptr, "%lf", parameters_ + i);
    }

    fclose(fptr);

    use_quaternions_ = use_quaternions;
    if (use_quaternions) {
        // 如果使用四元数表示相机位姿
        // Switch the angle-axis rotations to quaternions.
        num_parameters_ = 10 * num_cameras_ + 3 * num_points_;              // 使用四元数时参数的总数
        double *quaternion_parameters = new double[num_parameters_];        // 开辟存放四元数参数的数组
        double *original_cursor = parameters_;                              // 指向原来参数的指针赋给original_cusor
        double *quaternion_cursor = quaternion_parameters;                  // 指向新开辟的数组
        for (int i = 0; i < num_cameras_; ++i) {
            AngleAxisToQuaternion(original_cursor, quaternion_cursor);      // 将旋转向量转换为四元数
            quaternion_cursor += 4;                                         // 转换完之后游标向后移动四位，准备接受其他不变的参数
            original_cursor += 3;                                           // 相应的游标移动3位
            for (int j = 4; j < 10; ++j) {                                  // 观测数据10个参数，后面的包括平移向量和相机内参
                *quaternion_cursor++ = *original_cursor++;
            }
        }
        // 把路标点的参数copy到数组中
        for (int i = 0; i < 3 * num_points_; ++i) {
            *quaternion_cursor++ = *original_cursor++;
        }
        // Swap in the quaternion parameters.
        delete[]parameters_;   // 此时原先那个数组没有用了，要释放掉以免造成内存泄露
        parameters_ = quaternion_parameters;   // 再把新的数组指针赋给原来的指针，外界调用时就不用区别对待了
    }
}

void BALProblem::WriteToFile(const std::string &filename) const {
    FILE *fptr = fopen(filename.c_str(), "w");

    // 如果文件名为空则报错
    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filename;
        return;
    }

    // 向文件中写入数据，开头依次为 相机位姿数量、相机位姿数量、路标点数量、观测数据数量
    fprintf(fptr, "%d %d %d %d\n", num_cameras_, num_cameras_, num_points_, num_observations_);

    // 把剩余数据写入文件
    for (int i = 0; i < num_observations_; ++i) {
        fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
        for (int j = 0; j < 2; ++j) {
            fprintf(fptr, " %g", observations_[2 * i + j]);
        }
        fprintf(fptr, "\n");
    }

    for (int i = 0; i < num_cameras(); ++i) {
        double angleaxis[9];                    // 向文件中写入数据应当使用旋转向量格式
        if (use_quaternions_) {                 // 如果使用四元数，那么应当把其转换为旋转向量
            //OutPut in angle-axis format.
            QuaternionToAngleAxis(parameters_ + 10 * i, angleaxis);   // 转换函数
            memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));    // 剩余的参数直接copy
        } else {
            memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));    // 如果没有使用四元数，则可以直接参数都copy
        }
        // 把相机参数数据写盘
        for (int j = 0; j < 9; ++j) {
            fprintf(fptr, "%.16g\n", angleaxis[j]);
        }
    }

    // 把路标点的数据写盘
    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            fprintf(fptr, "%.16g\n", point[j]);
        }
    }

    fclose(fptr);
}

// 将问题写入PLY文件以在Meshlab或CloudCompare中进行检查
void BALProblem::WriteToPLYFile(const std::string &filename) const {
    std::ofstream of(filename.c_str());

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << num_cameras_ + num_points_
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    // Export extrinsic data (i.e. camera centers) as green points.
    // 输出外参数据作为绿色点
    double angle_axis[3];   // 欧拉角数组
    double center[3];       // 中心点
    for (int i = 0; i < num_cameras(); ++i) {
        const double *camera = cameras() + camera_block_size() * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        of << center[0] << ' ' << center[1] << ' ' << center[2]
           << " 0 255 0" << '\n';
    }

    // Export the structure (i.e. 3D Points) as white points.
    // 输出三维点作为白色点
    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            of << point[j] << ' ';
        }
        of << " 255 255 255\n";
    }
    of.close();
}

////// 将相机参数转换为欧拉角和中心点
void BALProblem::CameraToAngelAxisAndCenter(const double *camera,
                                            double *angle_axis,
                                            double *center) const {
    VectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        QuaternionToAngleAxis(camera, angle_axis);
    } else {
        angle_axis_ref = ConstVectorRef(camera, 3);
    }

    // c = -R't
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    AngleAxisRotatePoint(inverse_rotation.data(),
                         camera + camera_block_size() - 6,
                         center);
    VectorRef(center, 3) *= -1.0;
}

///// 将欧拉角以及中心点转换为相机参数
void BALProblem::AngleAxisAndCenterToCamera(const double *angle_axis,
                                            const double *center,
                                            double *camera) const {
    ConstVectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        AngleAxisToQuaternion(angle_axis, camera);
    } else {
        VectorRef(camera, 3) = angle_axis_ref;
    }

    // t = -R * c
    AngleAxisRotatePoint(angle_axis, center, camera + camera_block_size() - 6);
    VectorRef(camera + camera_block_size() - 6, 3) *= -1.0;
}

// 对读入数据进行归一化
void BALProblem::Normalize() {
    // Compute the marginal median of the geometry
    std::vector<double> tmp(num_points_);
    Eigen::Vector3d median;   // 中值
    double *points = mutable_points();  // 路标点数据的起始地址
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < num_points_; ++j) {
            tmp[j] = points[3 * j + i];
        }
        median(i) = Median(&tmp);  // 找到路标点坐标的中值
    }

    //  计算各个点和中值点的偏差并存入tmp[]数组中
    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        tmp[i] = (point - median).lpNorm<1>();
    }

    const double median_absolute_deviation = Median(&tmp);  // 偏差的中值

    // 缩放比例，以使最终重建的中值绝对偏差为100
    const double scale = 100.0 / median_absolute_deviation;

    // X = scale * (X - median)  对所有点的偏差进行缩放
    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        point = scale * (point - median);
    }

    double *cameras = mutable_cameras();
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = cameras + camera_block_size() * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        // center = scale * (center - median)
        VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);
        AngleAxisAndCenterToCamera(angle_axis, center, camera);
    }
}

////// 加入噪声
void BALProblem::Perturb(const double rotation_sigma,
                         const double translation_sigma,
                         const double point_sigma) {
    // 检验加在旋转向量、平移向量、路标点上的噪声sigma值均为正
    assert(point_sigma >= 0.0);
    assert(rotation_sigma >= 0.0);
    assert(translation_sigma >= 0.0);

    // 为路标点加入噪声
    double *points = mutable_points();
    if (point_sigma > 0) {
        for (int i = 0; i < num_points_; ++i) {
            PerturbPoint3(point_sigma, points + 3 * i);
        }
    }

    // 为相机参数加入噪声
    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = mutable_cameras() + camera_block_size() * i;

        double angle_axis[3];
        double center[3];
        // Perturb in the rotation of the camera in the angle-axis
        // representation
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        if (rotation_sigma > 0.0) {
            PerturbPoint3(rotation_sigma, angle_axis);
        }
        AngleAxisAndCenterToCamera(angle_axis, center, camera);

        if (translation_sigma > 0.0)
            PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
    }
}
