#pragma once

#include <stdint.h>
#include <string.h>
#include <iostream>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class IMUDriver {
   public:

    IMUDriver(){};
    virtual ~IMUDriver() = default;

    static std::shared_ptr<IMUDriver> create_imu(uint16_t imu_id, const std::string& interface_type, const std::string& interface,
                                                const std::string& imu_type, const int baudrate=0);

    virtual uint16_t get_imu_id() { return imu_id_; }
    virtual std::vector<float> get_ang_vel() { return ang_vel_; }
    virtual std::vector<float> get_quat() { return quat_; }
    virtual std::vector<float> get_lin_acc() { return lin_acc_; }
    virtual float get_temperature() { return temperature_; }

   protected:
    uint16_t imu_id_;

    std::vector<float> quat_{0.f, 0.f, 0.f, 0.f};       // w, x, y, z
    std::vector<float> ang_vel_{0.f, 0.f, 0.f};         // x, y, z
    std::vector<float> lin_acc_{0.f, 0.f, 0.f};         // x, y, z
    float temperature_{0.f}; // temperature
};