#pragma once

#include <cnpy.h>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

class MotionLoader {
public:
    MotionLoader(const std::string& motion_file);
    
    int get_fps() const { return fps_; }
    size_t get_num_frames() const { return num_frames_; }
    size_t get_num_joints() const { return num_joints_; }
    
    const std::vector<float>& get_pos(size_t frame) const {
        return joint_pos_[frame];
    }
    
    const std::vector<float>& get_vel(size_t frame) const {
        return joint_vel_[frame];
    }

private:
    int fps_;
    size_t num_frames_;
    size_t num_joints_;
    std::vector<std::vector<float>> joint_pos_;
    std::vector<std::vector<float>> joint_vel_;
};