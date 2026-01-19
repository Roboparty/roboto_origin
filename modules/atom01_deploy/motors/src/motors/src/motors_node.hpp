#pragma once
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <queue>
#include <motors/srv/control_motor.hpp>
#include <motors/srv/read_motors.hpp>
#include <motors/srv/reset_motors.hpp>
#include <motors/srv/set_zeros.hpp>
#include <motors/srv/clear_errors.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>

#include "close_chain_mapping.hpp"
#include "motor_driver.hpp"
#include "timer.hpp"

class MotorsNode : public rclcpp::Node {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MotorsNode() : Node("motors_node") {
        kp_.resize(12);
        kd_.resize(12);
        motors_model_.resize(12);
        joint_default_angle_.resize(23);
        ankle_decouple_ = std::make_shared<Decouple>();
        last_left_ankle_pos_(2);
        last_left_ankle_vel_(2);
        last_right_ankle_pos_(2);
        last_right_ankle_vel_(2);

        for (int i = 0; i < 4; ++i) {
            worker_threads_.emplace_back([this] {
                pthread_setname_np(pthread_self(), "thread_pool");
                struct sched_param sp{}; sp.sched_priority = 70;
                pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock,
                                              [this] { return this->stop_threads_.load() || !this->tasks_.empty(); });
                        if (this->stop_threads_.load() && this->tasks_.empty()) {
                            return;
                        }
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    if (task) {
                        task();
                    }
                }
            });
        }

        this->declare_parameter<std::string>("motors_type", "DM");
        this->declare_parameter<std::vector<int>>("motors_model",
                                                  std::vector<int>{1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0});
        this->declare_parameter<std::vector<float>>(
            "kp",
            std::vector<float>{100.0, 100.0, 150.0, 150.0, 40.0, 40.0, 150.0, 60.0, 60.0, 60.0, 40.0, 20.0});
        this->declare_parameter<std::vector<float>>(
            "kd", std::vector<float>{3.0, 3.0, 5.0, 5.0, 1.5, 1.5, 5.0, 2.0, 2.0, 2.0, 1.5, 1.0});
        this->declare_parameter<int>("can0_startID", 0);
        this->declare_parameter<int>("can0_endID", 0);
        this->declare_parameter<int>("can1_startID", 0);
        this->declare_parameter<int>("can1_endID", 0);
        this->declare_parameter<int>("can2_startID", 0);
        this->declare_parameter<int>("can2_endID", 0);
        this->declare_parameter<int>("can3_startID", 0);
        this->declare_parameter<int>("can3_endID", 0);
        this->declare_parameter<int>("can0_masterID_offset", 0);
        this->declare_parameter<int>("can1_masterID_offset", 0);
        this->declare_parameter<int>("can2_masterID_offset", 0);
        this->declare_parameter<int>("can3_masterID_offset", 0);
        this->declare_parameter<std::vector<float>>(
            "joint_default_angle",
            std::vector<float>{0.0, 0.0, -0.2, 0.4, -0.2, 0.0, 0.0, 0.0,   -0.2, 0.4, -0.2, 0.0,
                               0.0, 0.1, 0.07, 0.0, 1.0,  0.0, 0.1, -0.07, 0.0,  1.0, 0.0});

        std::vector<double> tmp;
        this->get_parameter("kp", tmp);
        std::transform(tmp.begin(), tmp.end(), kp_.begin(),
                       [](double val) { return static_cast<float>(val); });
        this->get_parameter("kd", tmp);
        std::transform(tmp.begin(), tmp.end(), kd_.begin(),
                       [](double val) { return static_cast<float>(val); });
        this->get_parameter("motors_type", motors_type_);
        std::vector<long int> tmp_ld;
        this->get_parameter("motors_model", tmp_ld);
        std::transform(tmp_ld.begin(), tmp_ld.end(), motors_model_.begin(),
                       [](long int val) { return static_cast<int>(val); });
        this->get_parameter("can0_startID", can0_startID_);
        this->get_parameter("can0_endID", can0_endID_);
        this->get_parameter("can1_startID", can1_startID_);
        this->get_parameter("can1_endID", can1_endID_);
        this->get_parameter("can2_startID", can2_startID_);
        this->get_parameter("can2_endID", can2_endID_);
        this->get_parameter("can3_startID", can3_startID_);
        this->get_parameter("can3_endID", can3_endID_);
        this->get_parameter("can0_masterID_offset", can0_masterID_offset_);
        this->get_parameter("can1_masterID_offset", can1_masterID_offset_);
        this->get_parameter("can2_masterID_offset", can2_masterID_offset_);
        this->get_parameter("can3_masterID_offset", can3_masterID_offset_);
        this->get_parameter("joint_default_angle", tmp);
        std::transform(tmp.begin(), tmp.end(), joint_default_angle_.begin(),
                       [](double val) { return static_cast<float>(val); });

        RCLCPP_INFO(this->get_logger(), "motors_type: %s", motors_type_.c_str());
        RCLCPP_INFO(this->get_logger(), "motors_model: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d",
                    motors_model_[0], motors_model_[1], motors_model_[2], motors_model_[3], motors_model_[4],
                    motors_model_[5], motors_model_[6], motors_model_[7], motors_model_[8], motors_model_[9],
                    motors_model_[10], motors_model_[11]);
        RCLCPP_INFO(this->get_logger(), "kp: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f", kp_[0], kp_[1],
                    kp_[2], kp_[3], kp_[4], kp_[5], kp_[6], kp_[7], kp_[8], kp_[9], kp_[10], kp_[11]);
        RCLCPP_INFO(this->get_logger(), "kd: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f", kd_[0], kd_[1],
                    kd_[2], kd_[3], kd_[4], kd_[5], kd_[6], kd_[7], kd_[8], kd_[9], kd_[10], kd_[11]);
        RCLCPP_INFO(this->get_logger(),
                    "joint_default_angle: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, "
                    "%f, %f, %f, %f, %f, %f, %f",
                    joint_default_angle_[0], joint_default_angle_[1], joint_default_angle_[2],
                    joint_default_angle_[3], joint_default_angle_[4], joint_default_angle_[5],
                    joint_default_angle_[6], joint_default_angle_[7], joint_default_angle_[8],
                    joint_default_angle_[9], joint_default_angle_[10], joint_default_angle_[11],
                    joint_default_angle_[12], joint_default_angle_[13], joint_default_angle_[14],
                    joint_default_angle_[15], joint_default_angle_[16], joint_default_angle_[17],
                    joint_default_angle_[18], joint_default_angle_[19], joint_default_angle_[20],
                    joint_default_angle_[21], joint_default_angle_[22]);
        RCLCPP_INFO(this->get_logger(), "can0_startID: %d", can0_startID_);
        RCLCPP_INFO(this->get_logger(), "can0_endID: %d", can0_endID_);
        RCLCPP_INFO(this->get_logger(), "can1_startID: %d", can1_startID_);
        RCLCPP_INFO(this->get_logger(), "can1_endID: %d", can1_endID_);
        RCLCPP_INFO(this->get_logger(), "can2_startID: %d", can2_startID_);
        RCLCPP_INFO(this->get_logger(), "can2_endID: %d", can2_endID_);
        RCLCPP_INFO(this->get_logger(), "can3_startID: %d", can3_startID_);
        RCLCPP_INFO(this->get_logger(), "can3_endID: %d", can3_endID_);
        RCLCPP_INFO(this->get_logger(), "can0_masterID_offset: %d", can0_masterID_offset_);
        RCLCPP_INFO(this->get_logger(), "can1_masterID_offset: %d", can1_masterID_offset_);
        RCLCPP_INFO(this->get_logger(), "can2_masterID_offset: %d", can2_masterID_offset_);
        RCLCPP_INFO(this->get_logger(), "can3_masterID_offset: %d", can3_masterID_offset_);

        left_leg_motors.resize(can0_endID_ - can0_startID_ + 1);
        right_leg_motors.resize(can1_endID_ - can1_startID_ + 1);
        left_arm_motors.resize(can2_endID_ - can2_startID_ + 1);
        right_arm_motors.resize(can3_endID_ - can3_startID_ + 1);

        left_ankle_motors_default_angle_.resize(2);
        right_ankle_motors_default_angle_.resize(2);
        left_ankle_motors_default_angle_ << joint_default_angle_[4], joint_default_angle_[5];
        right_ankle_motors_default_angle_ << joint_default_angle_[10], joint_default_angle_[11];
        Eigen::VectorXd vel = Eigen::VectorXd::Zero(2);
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(2);
        ankle_decouple_->getDecoupleQVT(left_ankle_motors_default_angle_, vel, tau, true);
        ankle_decouple_->getDecoupleQVT(right_ankle_motors_default_angle_, vel, tau, false);

        for (int i = can0_startID_; i <= can0_endID_; i++) {
            left_leg_motors[i - can0_startID_] = MotorDriver::MotorCreate(
                i, "can0", motors_type_, can0_masterID_offset_, motors_model_[i - can0_startID_]);
        }
        for (int i = can1_startID_; i <= can1_endID_; i++) {
            right_leg_motors[i - can1_startID_] = MotorDriver::MotorCreate(
                i, "can1", motors_type_, can1_masterID_offset_, motors_model_[i - can1_startID_]);
        }
        for (int i = can2_startID_; i <= can2_endID_; i++) {
            left_arm_motors[i - can2_startID_] = MotorDriver::MotorCreate(
                i, "can2", motors_type_, can2_masterID_offset_, motors_model_[i - can2_startID_ + 7]);
        }
        for (int i = can3_startID_; i <= can3_endID_; i++) {
            right_arm_motors[i - can3_startID_] = MotorDriver::MotorCreate(
                i, "can3", motors_type_, can3_masterID_offset_, motors_model_[i - can3_startID_ + 7]);
        }

        auto sensor_data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        auto control_command_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
        left_leg_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_command_left_leg", control_command_qos,
            std::bind(&MotorsNode::subs_left_leg_callback, this, std::placeholders::_1));
        right_leg_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_command_right_leg", control_command_qos,
            std::bind(&MotorsNode::subs_right_leg_callback, this, std::placeholders::_1));
        left_arm_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_command_left_arm", control_command_qos,
            std::bind(&MotorsNode::subs_left_arm_callback, this, std::placeholders::_1));
        right_arm_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_command_right_arm", control_command_qos,
            std::bind(&MotorsNode::subs_right_arm_callback, this, std::placeholders::_1));
        joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", sensor_data_qos, std::bind(&MotorsNode::subs_joy_callback, this, std::placeholders::_1));
        left_leg_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_states_left_leg", sensor_data_qos);
        right_leg_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_states_right_leg", sensor_data_qos);
        left_arm_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_states_left_arm", sensor_data_qos);
        right_arm_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_states_right_arm", sensor_data_qos);
        control_motor_service_ = this->create_service<motors::srv::ControlMotor>(
            "control_motor",
            std::bind(&MotorsNode::control_motor_srv, this, std::placeholders::_1, std::placeholders::_2));
        reset_motors_service_ = this->create_service<motors::srv::ResetMotors>(
            "reset_motors",
            std::bind(&MotorsNode::reset_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        read_motors_service_ = this->create_service<motors::srv::ReadMotors>(
            "read_motors",
            std::bind(&MotorsNode::read_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        set_zeros_service_ = this->create_service<motors::srv::SetZeros>(
            "set_zeros",
            std::bind(&MotorsNode::set_zeros_srv, this, std::placeholders::_1, std::placeholders::_2));
        clear_errors_service_ = this->create_service<motors::srv::ClearErrors>(
            "clear_errors",
            std::bind(&MotorsNode::clear_errors_srv, this, std::placeholders::_1, std::placeholders::_2));
    }
    ~MotorsNode() {
        if(is_init_){
            deinit_motors();
        }
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_threads_.store(true);
        }
        condition_.notify_all();
        for (std::thread &worker : worker_threads_) {
            worker.join();
        }
    }
    void publish_left_leg();
    void publish_right_leg();
    void publish_left_arm();
    void publish_right_arm();
    void subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg);
    void subs_left_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_right_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_left_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_right_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void reset_motors();
    void init_motors();
    void deinit_motors();
    void set_zeros();
    void read_motors();
    void clear_errors();
    void reset_motors_srv(const std::shared_ptr<motors::srv::ResetMotors::Request> request,
                      std::shared_ptr<motors::srv::ResetMotors::Response> response);
    void read_motors_srv(const std::shared_ptr<motors::srv::ReadMotors::Request> request,
                     std::shared_ptr<motors::srv::ReadMotors::Response> response);
    void set_zeros_srv(const std::shared_ptr<motors::srv::SetZeros::Request> request,
                   std::shared_ptr<motors::srv::SetZeros::Response> response);
    void control_motor_srv(const std::shared_ptr<motors::srv::ControlMotor::Request> request,
                               std::shared_ptr<motors::srv::ControlMotor::Response> response);
    void clear_errors_srv(const std::shared_ptr<motors::srv::ClearErrors::Request> request,
                          std::shared_ptr<motors::srv::ClearErrors::Response> response);

   private:
    std::atomic<bool> is_init_{false};
    int offline_threshold_ = 10;
    std::string motors_type_;
    std::vector<int> motors_model_;
    std::vector<std::shared_ptr<MotorDriver>> left_leg_motors, right_leg_motors, left_arm_motors,
        right_arm_motors;
    Eigen::VectorXd left_ankle_motors_default_angle_, right_ankle_motors_default_angle_;
    std::vector<float> kp_, kd_, joint_default_angle_;
    int can0_startID_, can0_endID_, can1_startID_, can1_endID_, can2_startID_, can2_endID_, can3_startID_,
        can3_endID_, can0_masterID_offset_, can1_masterID_offset_, can2_masterID_offset_,
        can3_masterID_offset_;
    std::shared_mutex left_leg_mutex_, right_leg_mutex_, left_arm_mutex_, right_arm_mutex_;
    rclcpp::Service<motors::srv::ControlMotor>::SharedPtr control_motor_service_;
    rclcpp::Service<motors::srv::ResetMotors>::SharedPtr reset_motors_service_;
    rclcpp::Service<motors::srv::ReadMotors>::SharedPtr read_motors_service_;
    rclcpp::Service<motors::srv::SetZeros>::SharedPtr set_zeros_service_;
    rclcpp::Service<motors::srv::ClearErrors>::SharedPtr clear_errors_service_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr left_leg_publisher_, right_leg_publisher_,
        left_arm_publisher_, right_arm_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_leg_subscription_,
        right_leg_subscription_, left_arm_subscription_, right_arm_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<Decouple> ankle_decouple_;
    Eigen::VectorXd last_left_ankle_pos_, last_left_ankle_vel_, last_right_ankle_pos_, last_right_ankle_vel_;
    int last_button0_ = 0, last_button1_ = 0;
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_threads_{false};
};