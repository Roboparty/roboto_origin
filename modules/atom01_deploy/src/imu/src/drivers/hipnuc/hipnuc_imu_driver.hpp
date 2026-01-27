#pragma once

extern "C" {
#include "hipnuc_dec.h"
#include "nmea_decode.h"
#include "hipnuc_can_common.h"
#include "hipnuc_j1939_parser.h"
#include "canopen_parser.h"
}

#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <shared_mutex>

#include "imu_driver.hpp"
#include "protocol/can/socket_can.hpp"
#include "protocol/serial/serial_port.hpp"

#define GRA_ACC     (9.8)
#define DEG_TO_RAD  (0.01745329)

class HipnucIMUDriver : public IMUDriver {
   public:
    HipnucIMUDriver(uint16_t imu_id, const std::string& interface_type, const std::string& interface, const int baudrate=0);
    ~HipnucIMUDriver();

    void can_rx_cbk(const can_frame& rx_frame);
    void serial_rx_cbk(const uint8_t* data, size_t length);
    std::vector<float> get_ang_vel() override;
    std::vector<float> get_quat() override;
    std::vector<float> get_lin_acc() override;
    float get_temperature() override;

   private:
    uint16_t imu_id_;
    int baudrate_;
    std::string interface_type_;
    std::string interface_;
    mutable std::shared_mutex imu_mutex_;
    std::shared_ptr<SocketCAN> can_;
    std::shared_ptr<SerialPort> serial_;
    can_sensor_data_t sensor_data_;
    hipnuc_raw_t raw_;
};