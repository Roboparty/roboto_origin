#include "motion_loader.hpp"

MotionLoader::MotionLoader(const std::string& motion_file) {
    try {
        // 加载 .npz 文件
        cnpy::npz_t npz_data = cnpy::npz_load(motion_file);
        
        // 读取 fps
        if (npz_data.find("fps") == npz_data.end()) {
            throw std::runtime_error("FPS not found in motion file");
        }
        cnpy::NpyArray fps_arr = npz_data["fps"];
        fps_ = *fps_arr.data<int>();
        
        // 读取 joint_pos
        if (npz_data.find("joint_pos") == npz_data.end()) {
            throw std::runtime_error("joint_pos not found in motion file");
        }
        cnpy::NpyArray joint_pos_arr = npz_data["joint_pos"];
        float* joint_pos_data = joint_pos_arr.data<float>();
        num_frames_ = joint_pos_arr.shape[0];
        num_joints_ = joint_pos_arr.shape[1];
        
        // 分配并填充 joint_pos
        joint_pos_.resize(num_frames_);
        for (size_t i = 0; i < num_frames_; ++i) {
            joint_pos_[i].resize(num_joints_);
            for (size_t j = 0; j < num_joints_; ++j) {
                joint_pos_[i][j] = joint_pos_data[i * num_joints_ + j];
            }
        }
        
        // 读取 joint_vel
        if (npz_data.find("joint_vel") == npz_data.end()) {
            throw std::runtime_error("joint_vel not found in motion file");
        }
        cnpy::NpyArray joint_vel_arr = npz_data["joint_vel"];
        float* joint_vel_data = joint_vel_arr.data<float>();
        
        // 分配并填充 joint_vel
        joint_vel_.resize(num_frames_);
        for (size_t i = 0; i < num_frames_; ++i) {
            joint_vel_[i].resize(num_joints_);
            for (size_t j = 0; j < num_joints_; ++j) {
                joint_vel_[i][j] = joint_vel_data[i * num_joints_ + j];
            }
        }
        
        std::cout << "Loaded motion file: " << motion_file << std::endl;
        std::cout << "FPS: " << fps_ << std::endl;
        std::cout << "Frames: " << num_frames_ << ", Joints: " << num_joints_ << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to load motion file: ") + e.what());
    }
}