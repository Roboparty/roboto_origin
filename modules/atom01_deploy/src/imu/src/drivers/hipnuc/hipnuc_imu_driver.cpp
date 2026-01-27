#include "hipnuc_imu_driver.hpp"

HipnucIMUDriver::HipnucIMUDriver(uint16_t imu_id, const std::string& interface_type, const std::string& interface, const int baudrate)
    : IMUDriver(), imu_id_(imu_id), interface_type_(interface_type), interface_(interface) {
    memset(&raw_, 0, sizeof(raw_));
    memset(&sensor_data_, 0, sizeof(sensor_data_));
    if (interface_type_ == "serial") {
        baudrate_ = baudrate;
        serial_ = SerialPort::open(interface_, baudrate_);
        SerialPort::SerialCbkFunc serial_callback = std::bind(&HipnucIMUDriver::serial_rx_cbk, this, std::placeholders::_1, std::placeholders::_2);
        serial_->set_serial_callback(serial_callback);
    } else if (interface_type_ == "can") {
        can_ = SocketCAN::get(interface_);
        CanCbkFunc can_callback = std::bind(&HipnucIMUDriver::can_rx_cbk, this, std::placeholders::_1);
        can_->add_can_callback(can_callback, imu_id_);
        can_->set_key_extractor([](const can_frame &frame) -> CanCbkId {
            return frame.can_id & 0x7F;
        });
    } else {
        throw std::runtime_error("Hipnuc driver only support CAN and SERIAL interface");
    }
}

HipnucIMUDriver::~HipnucIMUDriver() {
    if (interface_type_ == "serial" && serial_) {
        serial_->close();
    } else if (interface_type_ == "can" && can_) {
        can_->remove_can_callback(imu_id_);
    }
}

void HipnucIMUDriver::can_rx_cbk(const can_frame& rx_frame) {
    hipnuc_can_frame_t frame;
    frame.can_id = rx_frame.can_id;
    frame.can_dlc = rx_frame.can_dlc;
    memcpy(frame.data, rx_frame.data, 8);
    
    std::unique_lock<std::shared_mutex> lock(imu_mutex_);
    
    int ret = hipnuc_j1939_parse_frame(&frame, &sensor_data_);
    if (ret == 0) {
        return;
    }
    
    ret = canopen_parse_frame(&frame, &sensor_data_);
    if (ret == 0) {
        return;
    }
}

void HipnucIMUDriver::serial_rx_cbk(const uint8_t* data, size_t length) {
    std::unique_lock<std::shared_mutex> lock(imu_mutex_);
    
    for (size_t i = 0; i < length; i++) {
        if (hipnuc_input(&raw_, data[i])) {
            // Update sensor data from parsed raw data
            sensor_data_.quat_w = raw_.hi91.quat[0];
            sensor_data_.quat_x = raw_.hi91.quat[1];
            sensor_data_.quat_y = raw_.hi91.quat[2];
            sensor_data_.quat_z = raw_.hi91.quat[3];
            
            sensor_data_.gyr_x = raw_.hi91.gyr[0] * DEG_TO_RAD;
            sensor_data_.gyr_y = raw_.hi91.gyr[1] * DEG_TO_RAD;
            sensor_data_.gyr_z = raw_.hi91.gyr[2] * DEG_TO_RAD;
            
            sensor_data_.acc_x = raw_.hi91.acc[0] * GRA_ACC;
            sensor_data_.acc_y = raw_.hi91.acc[1] * GRA_ACC;
            sensor_data_.acc_z = raw_.hi91.acc[2] * GRA_ACC;
        }
    }
}

std::vector<float> HipnucIMUDriver::get_ang_vel() {
    std::shared_lock<std::shared_mutex> lock(imu_mutex_);
    return {sensor_data_.gyr_x, sensor_data_.gyr_y, sensor_data_.gyr_z};
}

std::vector<float> HipnucIMUDriver::get_quat() {
    std::shared_lock<std::shared_mutex> lock(imu_mutex_);
    return {sensor_data_.quat_w, sensor_data_.quat_x, sensor_data_.quat_y, sensor_data_.quat_z};
}

std::vector<float> HipnucIMUDriver::get_lin_acc() {
    std::shared_lock<std::shared_mutex> lock(imu_mutex_);
    return {sensor_data_.acc_x, sensor_data_.acc_y, sensor_data_.acc_z};
}

float HipnucIMUDriver::get_temperature() {
    std::shared_lock<std::shared_mutex> lock(imu_mutex_);
    return sensor_data_.temperature;
}