

#include "motor_driver.hpp"

#include "dm_motor_driver.hpp"

MotorDriver::MotorDriver() {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_st>());
    logger_ = setup_logger(sinks);
}
std::shared_ptr<MotorDriver> MotorDriver::MotorCreate(uint16_t motor_id, const char* interface,
                                                      const std::string motor_type, uint16_t master_id_offset,
                                                      int motor_model) {
    if (motor_type == "DM") {
        return std::make_shared<DmMotorDriver>(motor_id, interface, master_id_offset,
                                               static_cast<DM_Motor_Model>(motor_model));
    } else {
        throw std::runtime_error("Motor type not supported");
    }
}
