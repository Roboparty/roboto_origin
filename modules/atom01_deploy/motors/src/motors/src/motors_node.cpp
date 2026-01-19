#include "motors_node.hpp"

void MotorsNode::publish_left_leg() {
    Eigen::VectorXd q(2);
    q << left_leg_motors[4]->get_motor_pos(), left_leg_motors[5]->get_motor_pos();
    Eigen::VectorXd vel(2);
    vel << left_leg_motors[4]->get_motor_spd(), left_leg_motors[5]->get_motor_spd();
    Eigen::VectorXd tau(2);
    tau << left_leg_motors[4]->get_motor_current(), left_leg_motors[5]->get_motor_current();
    ankle_decouple_->getForwardQVT(q, vel, tau, true);
    last_left_ankle_pos_ = q;
    last_left_ankle_vel_ = vel;
    auto left_message = sensor_msgs::msg::JointState();
    left_message.header.stamp = this->now();
    left_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
    left_message.position = {left_leg_motors[0]->get_motor_pos() - joint_default_angle_[0],
                             left_leg_motors[1]->get_motor_pos() - joint_default_angle_[1],
                             left_leg_motors[2]->get_motor_pos() - joint_default_angle_[2],
                             left_leg_motors[3]->get_motor_pos() - joint_default_angle_[3],
                             q[0] - joint_default_angle_[4],
                             q[1] - joint_default_angle_[5]};
    left_message.velocity = {left_leg_motors[0]->get_motor_spd(),
                             left_leg_motors[1]->get_motor_spd(),
                             left_leg_motors[2]->get_motor_spd(),
                             left_leg_motors[3]->get_motor_spd(),
                             vel[0],
                             vel[1]};
    left_message.effort = {left_leg_motors[0]->get_motor_current(),
                           left_leg_motors[1]->get_motor_current(),
                           left_leg_motors[2]->get_motor_current(),
                           left_leg_motors[3]->get_motor_current(),
                           tau[0],
                           tau[1]};
    left_leg_publisher_->publish(left_message);
}

void MotorsNode::publish_right_leg() {
    Eigen::VectorXd q(2);
    q << -right_leg_motors[4]->get_motor_pos(), -right_leg_motors[5]->get_motor_pos();
    Eigen::VectorXd vel(2);
    vel << -right_leg_motors[4]->get_motor_spd(), -right_leg_motors[5]->get_motor_spd();
    Eigen::VectorXd tau(2);
    tau << -right_leg_motors[4]->get_motor_current(), -right_leg_motors[5]->get_motor_current();
    ankle_decouple_->getForwardQVT(q, vel, tau, false);
    last_right_ankle_pos_ = q;
    last_right_ankle_vel_ = vel;
    auto right_message = sensor_msgs::msg::JointState();
    right_message.header.stamp = this->now();
    right_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
    right_message.position = {right_leg_motors[0]->get_motor_pos() - joint_default_angle_[6],
                              right_leg_motors[1]->get_motor_pos() - joint_default_angle_[7],
                              -(right_leg_motors[2]->get_motor_pos() + joint_default_angle_[8]),
                              -(right_leg_motors[3]->get_motor_pos() + joint_default_angle_[9]),
                              q[0] - joint_default_angle_[10],
                              q[1] - joint_default_angle_[11],
                              right_leg_motors[6]->get_motor_pos() - joint_default_angle_[12]};
    right_message.velocity = {right_leg_motors[0]->get_motor_spd(),
                              right_leg_motors[1]->get_motor_spd(),
                              -right_leg_motors[2]->get_motor_spd(),
                              -right_leg_motors[3]->get_motor_spd(),
                              vel[0],
                              vel[1],
                              right_leg_motors[6]->get_motor_spd()};
    right_message.effort = {right_leg_motors[0]->get_motor_current(),
                            right_leg_motors[1]->get_motor_current(),
                            -right_leg_motors[2]->get_motor_current(),
                            -right_leg_motors[3]->get_motor_current(),
                            tau[0],
                            tau[1],
                            right_leg_motors[6]->get_motor_current()};
    right_leg_publisher_->publish(right_message);
}

void MotorsNode::publish_left_arm() {
    auto left_message = sensor_msgs::msg::JointState();
    left_message.header.stamp = this->now();
    left_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5"};
    left_message.position = {left_arm_motors[0]->get_motor_pos() - joint_default_angle_[13],
                             left_arm_motors[1]->get_motor_pos() - joint_default_angle_[14],
                             left_arm_motors[2]->get_motor_pos() - joint_default_angle_[15],
                             left_arm_motors[3]->get_motor_pos() - joint_default_angle_[16],
                             left_arm_motors[4]->get_motor_pos() - joint_default_angle_[17]};
    left_message.velocity = {left_arm_motors[0]->get_motor_spd(), left_arm_motors[1]->get_motor_spd(),
                             left_arm_motors[2]->get_motor_spd(), left_arm_motors[3]->get_motor_spd(),
                             left_arm_motors[4]->get_motor_spd()};
    left_message.effort = {left_arm_motors[0]->get_motor_current(), left_arm_motors[1]->get_motor_current(),
                           left_arm_motors[2]->get_motor_current(), left_arm_motors[3]->get_motor_current(),
                           left_arm_motors[4]->get_motor_current()};
    left_arm_publisher_->publish(left_message);
}

void MotorsNode::publish_right_arm() {
    auto right_message = sensor_msgs::msg::JointState();
    right_message.header.stamp = this->now();
    right_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5"};
    right_message.position = {-(right_arm_motors[0]->get_motor_pos() + joint_default_angle_[18]),
                              right_arm_motors[1]->get_motor_pos() - joint_default_angle_[19],
                              right_arm_motors[2]->get_motor_pos() - joint_default_angle_[20],
                              -(right_arm_motors[3]->get_motor_pos() + joint_default_angle_[21]),
                              right_arm_motors[4]->get_motor_pos() - joint_default_angle_[22]};
    right_message.velocity = {-right_arm_motors[0]->get_motor_spd(), right_arm_motors[1]->get_motor_spd(),
                              right_arm_motors[2]->get_motor_spd(), -right_arm_motors[3]->get_motor_spd(),
                              right_arm_motors[4]->get_motor_spd()};
    right_message.effort = {
        -right_arm_motors[0]->get_motor_current(), right_arm_motors[1]->get_motor_current(),
        right_arm_motors[2]->get_motor_current(), -right_arm_motors[3]->get_motor_current(),
        right_arm_motors[4]->get_motor_current()};
    right_arm_publisher_->publish(right_message);
}

void MotorsNode::subs_left_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    if(!is_init_.load()){
        RCLCPP_WARN(this->get_logger(), "Motors are not initialized.");
        return;
    }
    auto task = [this, msg]() {
        publish_left_leg();
        Eigen::VectorXd q(2);
        q << msg->position[4] + joint_default_angle_[4], msg->position[5] + joint_default_angle_[5];
        Eigen::VectorXd vel(2);
        vel << msg->velocity[4], msg->velocity[5];
        Eigen::VectorXd tau(2);
        tau << msg->effort[4] + kp_[4] * (q[0] - last_left_ankle_pos_[0]) +
                   kd_[4] * (vel[0] - last_left_ankle_vel_[0]),
            msg->effort[5] + kp_[5] * (q[1] - last_left_ankle_pos_[1]) +
                kd_[5] * (vel[1] - last_left_ankle_vel_[1]);
        ankle_decouple_->getDecoupleQVT(last_left_ankle_pos_, vel, tau, true);
        {
            std::unique_lock lock(left_leg_mutex_);
            for (int i = can0_startID_; i <= can0_endID_; i++) {
                if (i - can0_startID_ == 4 || i - can0_startID_ == 5) {
                    left_leg_motors[i - can0_startID_]->MotorMitModeCmd(q[i - can0_startID_ - 4],
                                                                        vel[i - can0_startID_ - 4], 0.0, 0.0,
                                                                        tau[i - can0_startID_ - 4]);
                } else {
                    left_leg_motors[i - can0_startID_]->MotorMitModeCmd(
                        msg->position[i - can0_startID_] + joint_default_angle_[i - can0_startID_],
                        msg->velocity[i - can0_startID_], kp_[i - can0_startID_], kd_[i - can0_startID_],
                        msg->effort[i - can0_startID_]);
                }
            Timer::ThreadSleepForUs(200);
                if(left_leg_motors[i - can0_startID_]->get_response_count() > offline_threshold_){
                    RCLCPP_ERROR(this->get_logger(), "Motor ID %d on CAN0 is offline!", i);
                    throw std::runtime_error("Motor offline, shutting down node.");
                }
            }
        }
    };
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.push(task);
    }
    condition_.notify_one();
}

void MotorsNode::subs_right_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    if(!is_init_.load()){
        RCLCPP_WARN(this->get_logger(), "Motors are not initialized.");
        return;
    }
    auto task = [this, msg]() {
        publish_right_leg();
        Eigen::VectorXd q(2);
        q << msg->position[4] + joint_default_angle_[10], msg->position[5] + joint_default_angle_[11];
        Eigen::VectorXd vel(2);
        vel << msg->velocity[4], msg->velocity[5];
        Eigen::VectorXd tau(2);
        tau << msg->effort[4] + kp_[4] * (q[0] - last_right_ankle_pos_[0]) +
                   kd_[4] * (vel[0] - last_right_ankle_vel_[0]),
            msg->effort[5] + kp_[5] * (q[1] - last_right_ankle_pos_[1]) +
                kd_[5] * (vel[1] - last_right_ankle_vel_[1]);
        ankle_decouple_->getDecoupleQVT(last_right_ankle_pos_, vel, tau, false);
        {
            std::unique_lock lock(right_leg_mutex_);
            for (int i = can1_startID_; i <= can1_endID_; i++) {
                if (i - can1_startID_ == 4 || i - can1_startID_ == 5) {
                    right_leg_motors[i - can1_startID_]->MotorMitModeCmd(-q[i - can1_startID_ - 4],
                                                                         -vel[i - can1_startID_ - 4], 0.0, 0.0,
                                                                         -tau[i - can1_startID_ - 4]);
                } else if (i - can1_startID_ == 2 || i - can1_startID_ == 3) {
                    right_leg_motors[i - can1_startID_]->MotorMitModeCmd(
                        -(msg->position[i - can1_startID_] + joint_default_angle_[i - can1_startID_ + 6]),
                        -msg->velocity[i - can1_startID_], kp_[i - can1_startID_], kd_[i - can1_startID_],
                        -msg->effort[i - can1_startID_]);
                } else {
                    right_leg_motors[i - can1_startID_]->MotorMitModeCmd(
                        msg->position[i - can1_startID_] + joint_default_angle_[i - can1_startID_ + 6],
                        msg->velocity[i - can1_startID_], kp_[i - can1_startID_], kd_[i - can1_startID_],
                        msg->effort[i - can1_startID_]);
                }
            Timer::ThreadSleepForUs(200);
                if(right_leg_motors[i - can1_startID_]->get_response_count() > offline_threshold_){
                    RCLCPP_ERROR(this->get_logger(), "Motor ID %d on CAN1 is offline!", i);
                    throw std::runtime_error("Motor offline, shutting down node.");
                }
            }
        }
    };
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.push(task);
    }
    condition_.notify_one();
}

void MotorsNode::subs_left_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    if(!is_init_.load()){
        RCLCPP_WARN(this->get_logger(), "Motors are not initialized.");
        return;
    }
    auto task = [this, msg](){
        publish_left_arm();
        std::unique_lock lock(left_arm_mutex_);
        for (int i = can2_startID_; i <= can2_endID_; i++) {
            left_arm_motors[i - can2_startID_]->MotorMitModeCmd(
                msg->position[i - can2_startID_] + joint_default_angle_[i - can2_startID_ + 13],
                msg->velocity[i - can2_startID_], kp_[i - can2_startID_ + 7], kd_[i - can2_startID_ + 7],
                msg->effort[i - can2_startID_]);
            Timer::ThreadSleepForUs(200);
            if(left_arm_motors[i - can2_startID_]->get_response_count() > offline_threshold_){
                RCLCPP_ERROR(this->get_logger(), "Motor ID %d on CAN2 is offline!", i);
                throw std::runtime_error("Motor offline, shutting down node.");
            }
        }
    };
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.push(task);
    }
    condition_.notify_one();
}

void MotorsNode::subs_right_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    if (!is_init_.load()) {
        RCLCPP_WARN(this->get_logger(), "Motors are not initialized.");
        return;
    }
    auto task = [this, msg](){
        publish_right_arm();
        std::unique_lock lock(right_arm_mutex_);
        for (int i = can3_startID_; i <= can3_endID_; i++) {
            if (i - can3_startID_ == 0 || i - can3_startID_ == 3) {
                right_arm_motors[i - can3_startID_]->MotorMitModeCmd(
                    -(msg->position[i - can3_startID_] + joint_default_angle_[i - can3_startID_ + 18]),
                    -msg->velocity[i - can3_startID_], kp_[i - can3_startID_ + 7], kd_[i - can3_startID_ + 7],
                    -msg->effort[i - can3_startID_]);
            } else {
                right_arm_motors[i - can3_startID_]->MotorMitModeCmd(
                    msg->position[i - can3_startID_] + joint_default_angle_[i - can3_startID_ + 18],
                    msg->velocity[i - can3_startID_], kp_[i - can3_startID_ + 7], kd_[i - can3_startID_ + 7],
                    msg->effort[i - can3_startID_]);
            }
            Timer::ThreadSleepForUs(200);
            if(right_arm_motors[i - can3_startID_]->get_response_count() > offline_threshold_){
                RCLCPP_ERROR(this->get_logger(), "Motor ID %d on CAN3 is offline!", i);
                throw std::runtime_error("Motor offline, shutting down node.");
            }
        }
    };
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.push(task);
    }
    condition_.notify_one();
}

void MotorsNode::subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg) {
    if (msg->buttons[0] == 1 and msg->buttons[0] != last_button0_) {
        if (!is_init_.load()){
            init_motors();
            RCLCPP_INFO(this->get_logger(), "Motors initialized.");
        }else{
            deinit_motors();
            RCLCPP_INFO(this->get_logger(), "Motors deinitialized.");
        }
    }
    if (msg->buttons[1] == 1 and msg->buttons[1] != last_button1_) {
        reset_motors();
        RCLCPP_INFO(this->get_logger(), "Motors reset.");
    }
    last_button0_ = msg->buttons[0];
    last_button1_ = msg->buttons[1];
}

void MotorsNode::init_motors() {
    std::scoped_lock lock(left_leg_mutex_, right_leg_mutex_, left_arm_mutex_, right_arm_mutex_);
    for (int i = can0_startID_; i <= can0_endID_; i++) {
        left_leg_motors[i - can0_startID_]->MotorInit();
        Timer::ThreadSleepForUs(200);
    }
    for (int i = can1_startID_; i <= can1_endID_; i++) {
        right_leg_motors[i - can1_startID_]->MotorInit();
        Timer::ThreadSleepForUs(200);
    }
    for (int i = can2_startID_; i <= can2_endID_; i++) {
        left_arm_motors[i - can2_startID_]->MotorInit();
        Timer::ThreadSleepForUs(200);
    }
    for (int i = can3_startID_; i <= can3_endID_; i++) {
        right_arm_motors[i - can3_startID_]->MotorInit();
        Timer::ThreadSleepForUs(200);
    }
    Timer::ThreadSleepFor(1000);
    publish_left_leg();
    publish_right_leg();
    publish_left_arm();
    publish_right_arm();
    is_init_.store(true);
}

void MotorsNode::deinit_motors() {
    std::scoped_lock lock(left_leg_mutex_, right_leg_mutex_, left_arm_mutex_, right_arm_mutex_);
    for (int i = can0_startID_; i <= can0_endID_; i++) {
        left_leg_motors[i - can0_startID_]->MotorDeInit();
        Timer::ThreadSleepForUs(200);
    }
    for (int i = can1_startID_; i <= can1_endID_; i++) {
        right_leg_motors[i - can1_startID_]->MotorDeInit();
        Timer::ThreadSleepForUs(200);
    }
    for (int i = can2_startID_; i <= can2_endID_; i++) {
        left_arm_motors[i - can2_startID_]->MotorDeInit();
        Timer::ThreadSleepForUs(200);
    }
    for (int i = can3_startID_; i <= can3_endID_; i++) {
        right_arm_motors[i - can3_startID_]->MotorDeInit();
        Timer::ThreadSleepForUs(200);
    }
    is_init_.store(false);
}

void MotorsNode::set_zeros() {
    if(!is_init_.load()){
        RCLCPP_WARN(this->get_logger(), "Motors are not initialized, cannot set zeros.");
        return;
    }
    {
        std::scoped_lock lock(left_leg_mutex_, right_leg_mutex_, left_arm_mutex_, right_arm_mutex_);
        for (int i = can0_startID_; i <= can0_endID_; i++) {
            left_leg_motors[i - can0_startID_]->MotorSetZero();
            Timer::ThreadSleepForUs(200);
        }
        for (int i = can1_startID_; i <= can1_endID_; i++) {
            right_leg_motors[i - can1_startID_]->MotorSetZero();
            Timer::ThreadSleepForUs(200);
        }
        for (int i = can2_startID_; i <= can2_endID_; i++) {
            left_arm_motors[i - can2_startID_]->MotorSetZero();
            Timer::ThreadSleepForUs(200);
        }
        for (int i = can3_startID_; i <= can3_endID_; i++) {
            right_arm_motors[i - can3_startID_]->MotorSetZero();
            Timer::ThreadSleepForUs(200);
        }
    }
}

void MotorsNode::reset_motors() {
    if(!is_init_.load()){
        RCLCPP_WARN(this->get_logger(), "Motors are not initialized, cannot reset.");
        return;
    }
    {
        std::scoped_lock lock(left_leg_mutex_, right_leg_mutex_, left_arm_mutex_, right_arm_mutex_);
        for (int i = can0_startID_; i <= can0_endID_; i++) {
            if (i - can0_startID_ == 4 || i - can0_startID_ == 5) {
                left_leg_motors[i - can0_startID_]->MotorMitModeCmd(
                    left_ankle_motors_default_angle_[i - can0_startID_ - 4], 0, 40, 2, 0);
            } else {
                left_leg_motors[i - can0_startID_]->MotorMitModeCmd(
                    joint_default_angle_[i - can0_startID_], 0, 40, 2, 0);
            }
            Timer::ThreadSleepForUs(200);
        }
        for (int i = can1_startID_; i <= can1_endID_; i++) {
            if (i - can1_startID_ == 4 || i - can1_startID_ == 5) {
                right_leg_motors[i - can1_startID_]->MotorMitModeCmd(
                    -right_ankle_motors_default_angle_[i - can1_startID_ - 4], 0, 40, 2, 0);
            } else if (i - can1_startID_ == 2 || i - can1_startID_ == 3) {
                right_leg_motors[i - can1_startID_]->MotorMitModeCmd(
                    -joint_default_angle_[i - can1_startID_ + 6], 0, 40, 2, 0);
            } else {
                right_leg_motors[i - can1_startID_]->MotorMitModeCmd(
                    joint_default_angle_[i - can1_startID_ + 6], 0, 40, 2, 0);
            }
            Timer::ThreadSleepForUs(200);
        }
        for (int i = can2_startID_; i <= can2_endID_; i++) {
            left_arm_motors[i - can2_startID_]->MotorMitModeCmd(
                joint_default_angle_[i - can2_startID_ + 13], 0, 40, 2, 0);
                Timer::ThreadSleepForUs(200);
        }
        for (int i = can3_startID_; i <= can3_endID_; i++) {
            if (i - can3_startID_ == 0 || i - can3_startID_ == 3) {
                right_arm_motors[i - can3_startID_]->MotorMitModeCmd(
                    -joint_default_angle_[i - can3_startID_ + 18], 0, 40, 2, 0);
            } else {
                right_arm_motors[i - can3_startID_]->MotorMitModeCmd(
                    joint_default_angle_[i - can3_startID_ + 18], 0, 40, 2, 0);
            }
            Timer::ThreadSleepForUs(200);
        }
        Timer::ThreadSleepFor(1000);
        for (int i = can0_startID_; i <= can0_endID_; i++) {
            if (i - can0_startID_ == 4 || i - can0_startID_ == 5) {
                left_leg_motors[i - can0_startID_]->MotorMitModeCmd(
                    left_ankle_motors_default_angle_[i - can0_startID_ - 4], 0, kp_[i - can0_startID_],
                    kd_[i - can0_startID_], 0);
            } else {
                left_leg_motors[i - can0_startID_]->MotorMitModeCmd(
                    joint_default_angle_[i - can0_startID_], 0, kp_[i - can0_startID_],
                    kd_[i - can0_startID_], 0);
            }
            Timer::ThreadSleepForUs(200);
        }
        for (int i = can1_startID_; i <= can1_endID_; i++) {
            if (i - can1_startID_ == 4 || i - can1_startID_ == 5) {
                right_leg_motors[i - can1_startID_]->MotorMitModeCmd(
                    -right_ankle_motors_default_angle_[i - can1_startID_ - 4], 0, kp_[i - can1_startID_],
                    kd_[i - can1_startID_], 0);
            } else if (i - can1_startID_ == 2 || i - can1_startID_ == 3) {
                right_leg_motors[i - can1_startID_]->MotorMitModeCmd(
                    -joint_default_angle_[i - can1_startID_ + 6], 0, kp_[i - can1_startID_],
                    kd_[i - can1_startID_], 0);
            } else {
                right_leg_motors[i - can1_startID_]->MotorMitModeCmd(
                    joint_default_angle_[i - can1_startID_ + 6], 0, kp_[i - can1_startID_],
                    kd_[i - can1_startID_], 0);
            }
            Timer::ThreadSleepForUs(200);
        }
        for (int i = can2_startID_; i <= can2_endID_; i++) {
            left_arm_motors[i - can2_startID_]->MotorMitModeCmd(
                joint_default_angle_[i - can2_startID_ + 13], 0, kp_[i - can2_startID_ + 7],
                kd_[i - can2_startID_ + 7], 0);
                Timer::ThreadSleepForUs(200);
        }
        for (int i = can3_startID_; i <= can3_endID_; i++) {
            if (i - can3_startID_ == 0 || i - can3_startID_ == 3) {
                right_arm_motors[i - can3_startID_]->MotorMitModeCmd(
                    -joint_default_angle_[i - can3_startID_ + 18], 0, kp_[i - can3_startID_ + 7],
                    kd_[i - can3_startID_ + 7], 0);
            } else {
                right_arm_motors[i - can3_startID_]->MotorMitModeCmd(
                    joint_default_angle_[i - can3_startID_ + 18], 0, kp_[i - can3_startID_ + 7],
                    kd_[i - can3_startID_ + 7], 0);
            }
            Timer::ThreadSleepForUs(200);
        }
    }
    Timer::ThreadSleepFor(1000);
    publish_left_leg();
    publish_right_leg();
    publish_left_arm();
    publish_right_arm();
}

void MotorsNode::read_motors(){
    auto read_left_leg_task = [this]() {
        std::unique_lock lock(left_leg_mutex_);
        for (int i = can0_startID_; i <= can0_endID_; i++) {
            left_leg_motors[i - can0_startID_]->refresh_motor_status();
            Timer::ThreadSleepForUs(200);
        }
        Timer::ThreadSleepForUs(1000);
        publish_left_leg();
    };

    auto read_right_leg_task = [this]() {
        std::unique_lock lock(right_leg_mutex_);
        for (int i = can1_startID_; i <= can1_endID_; i++) {
            right_leg_motors[i - can1_startID_]->refresh_motor_status();
            Timer::ThreadSleepForUs(200);
        }
        Timer::ThreadSleepForUs(1000);
        publish_right_leg();
    };

    auto read_left_arm_task = [this]() {
        std::unique_lock lock(left_arm_mutex_);
        for (int i = can2_startID_; i <= can2_endID_; i++) {
            left_arm_motors[i - can2_startID_]->refresh_motor_status();
            Timer::ThreadSleepForUs(200);
        }
        Timer::ThreadSleepForUs(1000);
        publish_left_arm();
    };

    auto read_right_arm_task = [this]() {
        std::unique_lock lock(right_arm_mutex_);
        for (int i = can3_startID_; i <= can3_endID_; i++) {
            right_arm_motors[i - can3_startID_]->refresh_motor_status();
            Timer::ThreadSleepForUs(200);
        }
        Timer::ThreadSleepForUs(1000);
        publish_right_arm();
    };

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.push(read_left_leg_task);
        tasks_.push(read_right_leg_task);
        tasks_.push(read_left_arm_task);
        tasks_.push(read_right_arm_task);
    }
    condition_.notify_all();
}

void MotorsNode::clear_errors() {
    std::scoped_lock lock(left_leg_mutex_, right_leg_mutex_, left_arm_mutex_, right_arm_mutex_);
    for (int i = can0_startID_; i <= can0_endID_; i++) {
        left_leg_motors[i - can0_startID_]->clear_motor_error();
        Timer::ThreadSleepForUs(200);
    }
    for (int i = can1_startID_; i <= can1_endID_; i++) {
        right_leg_motors[i - can1_startID_]->clear_motor_error();
        Timer::ThreadSleepForUs(200);
    }
    for (int i = can2_startID_; i <= can2_endID_; i++) {
        left_arm_motors[i - can2_startID_]->clear_motor_error();
        Timer::ThreadSleepForUs(200);
    }
    for (int i = can3_startID_; i <= can3_endID_; i++) {
        right_arm_motors[i - can3_startID_]->clear_motor_error();
        Timer::ThreadSleepForUs(200);
    }
}

void MotorsNode::reset_motors_srv(const std::shared_ptr<motors::srv::ResetMotors::Request> request,
                              std::shared_ptr<motors::srv::ResetMotors::Response> response) {
    if(!is_init_.load()){
        response->success = false;
        response->message = "Motors are not initialized, cannot reset motors.";
        return;
    }
    try {
        reset_motors();
        response->success = true;
        response->message = "Motors reset successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void MotorsNode::read_motors_srv(const std::shared_ptr<motors::srv::ReadMotors::Request> request,
                             std::shared_ptr<motors::srv::ReadMotors::Response> response) {
    if(!is_init_.load()){
        response->success = false;
        response->message = "Motors are not initialized, cannot read motors.";
        return;
    }
    try {
        read_motors();
        response->success = true;
        response->message = "Motors read successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void MotorsNode::set_zeros_srv(const std::shared_ptr<motors::srv::SetZeros::Request> request,
                           std::shared_ptr<motors::srv::SetZeros::Response> response) {
    if(!is_init_.load()){
        response->success = false;
        response->message = "Motors are not initialized, cannot set zeros.";
        return;
    }
    try {
        set_zeros();
        response->success = true;
        response->message = "Zeros set successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void MotorsNode::control_motor_srv(const std::shared_ptr<motors::srv::ControlMotor::Request> request,
                               std::shared_ptr<motors::srv::ControlMotor::Response> response) {
    int motor_id = request->motor_id;
    int can_id = request->can_id;
    float position = request->position;
    float velocity = request->velocity;
    float effort = request->effort;
    if(!is_init_.load()){
        response->success = false;
        response->message = "Motors are not initialized, cannot control motor.";
        return;
    }
    try {
        if (can_id == 0 && motor_id >= can0_startID_ && motor_id <= can0_endID_) {
            std::unique_lock lock(left_leg_mutex_);
            left_leg_motors[motor_id - can0_startID_]->MotorMitModeCmd(
                position, velocity, kp_[motor_id - can0_startID_], kd_[motor_id - can0_startID_], effort);
            response->success = true;
            response->message = "Send sucessfully";
        } else if (can_id == 1 && motor_id >= can1_startID_ && motor_id <= can1_endID_) {
            std::unique_lock lock(right_leg_mutex_);
            right_leg_motors[motor_id - can1_startID_]->MotorMitModeCmd(
                position, velocity, kp_[motor_id - can1_startID_], kd_[motor_id - can1_startID_], effort);
            response->success = true;
            response->message = "Send sucessfully";
        } else if (can_id == 2 && motor_id >= can2_startID_ && motor_id <= can2_endID_) {
            std::unique_lock lock(left_arm_mutex_);
            left_arm_motors[motor_id - can2_startID_]->MotorMitModeCmd(
                position, velocity, kp_[motor_id - can2_startID_ + 7], kd_[motor_id - can2_startID_ + 7],
                effort);
            response->success = true;
            response->message = "Send sucessfully";
        } else if (can_id == 3 && motor_id >= can3_startID_ && motor_id <= can3_endID_) {
            std::unique_lock lock(right_arm_mutex_);
            right_arm_motors[motor_id - can3_startID_]->MotorMitModeCmd(
                position, velocity, kp_[motor_id - can3_startID_ + 7], kd_[motor_id - can3_startID_ + 7],
                effort);
            response->success = true;
            response->message = "Send sucessfully";
        } else {
            response->success = false;
            response->message = "Invalid motor ID";
        }
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    }
}

void MotorsNode::clear_errors_srv(const std::shared_ptr<motors::srv::ClearErrors::Request> request,
                           std::shared_ptr<motors::srv::ClearErrors::Response> response) {
    try {
        clear_errors();
        response->success = true;
        response->message = "Clear errors successfully";
    } catch (const std::exception& e) {
        response->success = false;
        response->message = e.what();
    } 
}
    

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MotorsNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    RCLCPP_INFO(node->get_logger(), "Press 'X' to initialize/deinitialize motors.");
    RCLCPP_INFO(node->get_logger(), "Press 'A' to reset motors.");
    RCLCPP_INFO(node->get_logger(), "Press 'Y' to read motors.");
    try {
        executor.spin();
    } catch (const std::runtime_error& e) {
        node->deinit_motors();
        RCLCPP_FATAL(node->get_logger(), "Caught exception: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}