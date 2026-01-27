#include "motor_driver.hpp"

#include "dm_motor_driver.hpp"
#include "evo_motor_driver.hpp"

MotorDriver::MotorDriver() {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_st>());
    logger_ = setup_logger(sinks);
}
std::shared_ptr<MotorDriver> MotorDriver::create_motor(uint16_t motor_id, const std::string& interface_type, const std::string& interface,
                                                      const std::string& motor_type, int motor_model, uint16_t master_id_offset) {
    if (motor_type == "DM") {
        return std::make_shared<DmMotorDriver>(motor_id, interface_type, interface, master_id_offset,
                                               static_cast<DM_Motor_Model>(motor_model));
    } else if (motor_type == "EVO") { 
        return std::make_shared<EvoMotorDriver>(motor_id, interface_type, interface,
                                                static_cast<EVO_Motor_Model>(motor_model));
    } else {
        throw std::runtime_error("Motor type not supported");
    }
}
