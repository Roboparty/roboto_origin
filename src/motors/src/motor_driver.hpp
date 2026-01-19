#pragma once
// #include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <stdint.h>
#include <string.h>

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "SocketCAN.hpp"
#include "iostream"
#include "timer.hpp"
#include "utils.hpp"
using std::vector;
class MotorDriver {
   public:
    enum MotorControlMode_e {
        NONE = 0,
        MIT = 1,
        POS = 2,
        SPD = 3,
    };

    static constexpr double judgment_accuracy_threshold = 1e-2;
    static const int normal_sleep_time = 5;
    static const int setup_sleep_time = 500;

    MotorDriver();
    virtual ~MotorDriver() = default;

    static std::shared_ptr<MotorDriver> MotorCreate(uint16_t motor_id, const char* interface,
                                                    const std::string motor_type, uint16_t master_id_offset,
                                                    const int motor_model);

    virtual void CanRxMsgCallback(const can_frame& rx_frame) = 0;
    /**
     * @brief Locks the motor to prevent movement.
     *
     * This function locks the motor to prevent any movement.
     * Once locked, the motor will not respond to commands for movement.
     */
    virtual void MotorLock() = 0;

    /**
     * @brief Unlocks the motor to allow movement.
     *
     * This function unlocks the motor to enable movement.
     * After unlocking, the motor can respond to movement commands.
     */
    virtual void MotorUnlock() = 0;

    /**
     * @brief Initializes the motor.
     *
     * This function initializes the motor for operation.
     * It performs necessary setup and configuration for motor control.
     *
     * @return True if motor initialization is successful; otherwise, false.
     */
    virtual uint8_t MotorInit() = 0;

    /**
     * @brief Deinitializes the motor.
     *
     * This function deinitializes the motor.
     * It performs cleanup and releases resources associated with motor control.
     */
    virtual void MotorDeInit() = 0;

    /**
     * @brief Sets the motor position to zero.
     *
     * This function sets the current motor position to zero.
     * It establishes a new reference point for position measurement.
     *
     * @return True if setting motor position to zero is successful; otherwise, false.
     */
    virtual bool MotorSetZero() = 0;

    virtual bool MotorWriteFlash() = 0;

    /**
     * @brief Requests motor parameters based on a specific command.
     *
     * This function sends a request to retrieve specific parameters from the motor.
     * The parameter to be retrieved is identified by the `param_cmd` argument.
     *
     * @param param_cmd The command code specifying which parameter to retrieve.
     */
    virtual void MotorGetParam(uint8_t param_cmd) = 0;
    // to enum and union

    /**
     * @brief Commands the motor to move to a specified position at a specified speed.
     *
     * This function is responsible for commanding the motor to move to a desired position
     * with a specified speed.
     *
     * @param pos The target position to move the motor to.
     * @param spd The speed at which the motor should move to the target position.
     * @param ignore_limit If true, ignores any position limits that may be set.
     */
    virtual void MotorPosModeCmd(float pos, float spd, bool ignore_limit = false) = 0;

    /**
     * @brief Commands the motor to rotate at a specified speed.
     *
     * This function commands the motor to rotate at the specified speed.
     *
     * @param spd The speed at which the motor should rotate.
     */
    virtual void MotorSpdModeCmd(float spd) = 0;

    /**
     * @brief Commands the motor to operate in impedance mode with specific parameters.
     *
     * This function sets the motor to operate in an impedance control mode, where
     * it applies force based on the provided parameters.
     *
     * @param f_p Proportional force value.
     * @param f_v Velocity-based force value.
     * @param f_kp Proportional stiffness coefficient.
     * @param f_kd Damping coefficient.
     * @param f_t Desired torque value.
     */
    virtual void MotorMitModeCmd(float f_p, float f_v, float f_kp, float f_kd, float f_t) = 0;

    /**
     * @brief Sets the control mode for the motor.
     *
     * This function specifies the control mode for the motor.
     *
     * @param motor_control_mode The control mode to be set for the motor.
     */
    virtual void set_motor_control_mode(uint8_t motor_control_mode) = 0;

    /**
     * @brief Retrieves the count of responses received from the motor.
     *
     * This function returns the number of responses received from the motor.
     *
     * @return The count of responses received from the motor.
     */
    virtual int get_response_count() const = 0;
    virtual void refresh_motor_status() = 0;

    // to get torque
    // to get error code

    virtual void MotorResetID() = 0;

    /**
     * @brief Retrieves the ID of the motor.
     *
     * This function returns the unique identifier (ID) of the motor.
     *
     * @return The ID of the motor.
     */
    virtual uint8_t get_motor_id() { return motor_id_; }

    /**
     * @brief Retrieves the control mode of the motor.
     *
     * This function returns the current control mode of the motor.
     *
     * @return The control mode of the motor.
     */
    virtual uint8_t get_motor_control_mode() { return motor_control_mode_; }

    /**
     * @brief Retrieves the error ID associated with the motor.
     *
     * This function returns the error ID that indicates any error condition of the motor.
     *
     * @return The error ID of the motor.
     */
    virtual uint8_t get_error_id() { return error_id_; }

    /**
     * @brief Retrieves the current position of the motor.
     *
     * This function returns the current position of the motor.
     *
     * @return The current position of the motor.
     */
    virtual float get_motor_pos() { return motor_pos_; }

    /**
     * @brief Retrieves the current speed of the motor.
     *
     * This function returns the current speed of the motor.
     *
     * @return The current speed of the motor.
     */
    virtual float get_motor_spd() { return motor_spd_; }

    /**
     * @brief Retrieves the current current (electric current) of the motor.
     *
     * This function returns the electric current flowing through the motor.
     *
     * @return The current (electric current) of the motor.
     */
    virtual float get_motor_current() { return motor_current_; }

    /**
     * @brief Retrieves the temperature of the motor.
     *
     * This function returns the current temperature of the motor.
     *
     * @return The temperature of the motor.
     */
    virtual float get_motor_temperature() { return motor_temperature_; }

    virtual void clear_motor_error() = 0;

   protected:
    std::shared_ptr<spdlog::logger> logger_;
    uint16_t motor_id_;
    uint16_t master_id_;

    uint8_t motor_control_mode_;  // 0:none 1:pos 2:spd 3:mit

    std::atomic<uint8_t> error_id_{0};

    std::atomic<float> motor_pos_{0.f};
    std::atomic<float> motor_spd_{0.f};
    std::atomic<float> motor_current_{0.f};
    std::atomic<float> motor_temperature_{0.f};
};

using union32_t = union Union32 {
    float f;
    int32_t i;
    uint32_t u;
    uint8_t buf[4];
};
