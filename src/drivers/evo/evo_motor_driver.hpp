#pragma once

#include <atomic>
#include <string>

#include "motor_driver.hpp"
#include "protocol/can/socket_can.hpp"
enum EVOError {
    EVO_NO_ERROR = 0x00,
    EVO_OVER_VOLTAGE = 0x01,
    EVO_UNDER_VOLTAGE = 0x02,
    EVO_OVER_CURRENT = 0x03,
    EVO_MOS_OVER_TEMP = 0x04,
    EVO_COIL_OVER_TEMP = 0x05,
    EVO_COMM_LOST = 0x06,
    EVO_OVERLOAD = 0x07,
    EVO_ENCODER_ERROR = 0x08,
    EVO_UNKNOWN_ERROR = 0xFF
};

enum EVO_Motor_Model { 
    EVO431040 = 0,
    EVO811825 = 1,
    EVO811832 = 2,
    EVO_Num_Of_Model 
};

enum EVO_REG {
    EVO_CMD_MOTOR_MODE = 0xFC,      ///< Enable motor mode
    EVO_CMD_RESET_MODE = 0xFD,      ///< Reset motor and clear errors
    EVO_CMD_SET_ZERO = 0xFE,        ///< Set current position as zero point
    EVO_CMD_WRITE_FLASH = 0x12,     ///< Write parameter to flash memory
    EVO_CMD_READ_FLASH = 0x13,      ///< Read parameters from flash memory
    EVO_CMD_REBOOT = 0xFE           ///< Reboot motor (make flash parameters effective)
};

enum EVO_Flash_Param {
    EVO_PARAM_Q_MAX = 0x00,         ///< Maximum position limit
    EVO_PARAM_Q_MIN = 0x01,         ///< Minimum position limit
    EVO_PARAM_DQ_MAX = 0x02,        ///< Maximum velocity limit
    EVO_PARAM_DQ_MIN = 0x03,        ///< Minimum velocity limit
    EVO_PARAM_TAU_MAX = 0x04,       ///< Maximum torque/current limit
    EVO_PARAM_TAU_MIN = 0x05,       ///< Minimum torque/current limit
    EVO_PARAM_OKP_MAX = 0x06,       ///< Maximum outer Kp
    EVO_PARAM_OKP_MIN = 0x07,       ///< Minimum outer Kp
    EVO_PARAM_OKD_MAX = 0x08,       ///< Maximum outer Kd
    EVO_PARAM_OKD_MIN = 0x09,       ///< Minimum outer Kd
    EVO_PARAM_IKP_MAX = 0x0A,       ///< Maximum inner Kp
    EVO_PARAM_IKP_MIN = 0x0B,       ///< Minimum inner Kp
    EVO_PARAM_IKI_MAX = 0x0C,       ///< Maximum inner Ki
    EVO_PARAM_IKI_MIN = 0x0D,       ///< Minimum inner Ki
    EVO_PARAM_CUR_MAX = 0x0E,       ///< Maximum current
    EVO_PARAM_CUR_MIN = 0x0F        ///< Minimum current
};

typedef struct {
    float PosMax;       ///< Maximum position limit (rad)
    float SpdMax;       ///< Maximum velocity limit (rad/s)
    float TauMax;       ///< Maximum torque limit (N·m)
    float OKpMax;       ///< Maximum outer-loop proportional gain
    float OKdMax;       ///< Maximum outer-loop derivative gain
} EVO_Limit_Param;

class EvoMotorDriver : public MotorDriver {
   public:
    EvoMotorDriver(uint16_t motor_id, const std::string& interface_type, const std::string& can_interface,
                   EVO_Motor_Model motor_model);
    ~EvoMotorDriver();

    virtual void lock_motor() override;
    virtual void unlock_motor() override;
    virtual uint8_t init_motor() override;
    virtual void deinit_motor() override;
    virtual bool set_motor_zero() override;
    virtual bool write_motor_flash() override;

    virtual void get_motor_param(uint8_t param_cmd) override;
    virtual void motor_pos_cmd(float pos, float spd, bool ignore_limit) override {};
    virtual void motor_spd_cmd(float spd) override {};
    virtual void motor_mit_cmd(float f_p, float f_v, float f_kp, float f_kd, float f_t) override;
    virtual void reset_motor_id() override {};
    virtual void set_motor_control_mode(uint8_t motor_control_mode) override;
    virtual int get_response_count() const { 
        return response_count_; 
    }
    virtual void refresh_motor_status() override;
    virtual void clear_motor_error() override;
   private:
    std::atomic<int> response_count_{0};
    EVO_Motor_Model motor_model_;
    EVO_Limit_Param limit_param_;
    std::atomic<uint8_t> mos_temperature_{0};
    void set_motor_zero_evo();
    void clear_motor_error_evo();
    void write_register_evo(uint16_t index, uint8_t subindex, int32_t value);
    void read_register_evo(uint16_t index, uint8_t subindex);
    void save_register_evo(uint8_t rid);
    virtual void can_rx_cbk(const can_frame& rx_frame);
    std::shared_ptr<SocketCAN> can_;
};