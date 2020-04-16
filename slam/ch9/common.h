#pragma once   // 由编译器提供保证：同一个文件不会被包含多次;功能同 #ifndef

/// 从文件读入BAL dataset
class BALProblem {
public:
    /// load bal data from text file
    explicit BALProblem(const std::string &filename, bool use_quaternions = false);

    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

/////// 下面这些函数 先声明一下，具体要怎么实现以及实现什么程度的效果由使用者自己在common.cpp中定义 ////

    /// 存储结果到文件中
    void WriteToFile(const std::string &filename) const;

    // 把结果存储到点云数据中
    void WriteToPLYFile(const std::string &filename) const;

    // 对原始数据进行归一化
    void Normalize();

    ////// 给原始数据加上噪声
    void Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);

    // 相机位姿参数的维度（若使用四元数则为10，不使用四元数则为9）
    int camera_block_size() const { return use_quaternions_ ? 10 : 9; }

    // 路标点参数的维度 ： 3
    int point_block_size() const { return 3; }

    // 观测数据中 相机位姿点的数量
    int num_cameras() const { return num_cameras_; }

    // 观测数据中 路标点的数量
    int num_points() const { return num_points_; }

    // 观测数据的数量
    int num_observations() const { return num_observations_; }

    // 参数的数量
    int num_parameters() const { return num_parameters_; }

    // 路标点起始地址
    const int *point_index() const { return point_index_; }

    // 相机位姿起始地址
    const int *camera_index() const { return camera_index_; }

    // 观测数据起始地址
    const double *observations() const { return observations_; }

    // 相机参数的起始地址
    const double *parameters() const { return parameters_; }

    // 相机参数的起始地址
    const double *cameras() const { return parameters_; }

    //路标点的起始地址
    const double *points() const { return parameters_ + camera_block_size() * num_cameras_; }

    /// camera参数的起始地址
    double *mutable_cameras() { return parameters_; }

    /// 路标点的起始地址
    double *mutable_points() { return parameters_ + camera_block_size() * num_cameras_; }

    // 第i个相机位姿观测数据值的地址
    double *mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }

    // 第i个路标点位姿观测数据值的地址
    double *mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * point_block_size();
    }

    // 第i个相机位姿观测数据值的地址
    const double *camera_for_observation(int i) const {
        return cameras() + camera_index_[i] * camera_block_size();
    }

    // 第i个路标点位姿观测数据值的地址
    const double *point_for_observation(int i) const {
        return points() + point_index_[i] * point_block_size();
    }

private:
    void CameraToAngelAxisAndCenter(const double *camera,
                                    double *angle_axis,
                                    double *center) const;

    void AngleAxisAndCenterToCamera(const double *angle_axis,
                                    const double *center,
                                    double *camera) const;

    int num_cameras_;       // 相机位姿的数量
    int num_points_;        // 路标点的数量
    int num_observations_;  // 观测数据的数量
    int num_parameters_;
    bool use_quaternions_;  // 是否使用四元数

    int *point_index_;      // 每个observation对应的point index
    int *camera_index_;     // 每个observation对应的camera index
    double *observations_;  // 观测数据数组
    double *parameters_;    // 相机参数数组
};
