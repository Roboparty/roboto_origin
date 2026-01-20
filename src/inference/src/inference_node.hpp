#include <onnxruntime_cxx_api.h>
#include <string.h>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp> 
#include "motion_loader.hpp"

class InferenceNode : public rclcpp::Node {
   public:
    InferenceNode() : Node("inference_node") {
        gravity_z_upper_ = -0.7;
        joint_limits_lower_ = std::vector<float>{-1.0,-0.6,-1.57+0.1,-0.2-0.3,-1.0+0.2,-0.5,-0.6,-1.0,-1.57+0.1,-0.2-0.3,-1.0+0.2,-0.5,-3.14,-3.14-0.18,-0.25-0.06,-1.57,-0.6-0.78,-1.57,-3.14-0.18,-3.14+0.06,-1.57,-0.6-0.78,-1.57};
        joint_limits_upper_ = std::vector<float>{ 0.6, 1.0, 1.57+0.1, 2.5-0.3, 1.0+0.2, 0.5, 1.0, 0.6, 1.57+0.1, 2.5-0.3, 1.0+0.2, 0.5, 3.14, 1.57-0.18, 3.14-0.06, 1.57,1.57-0.78, 1.57, 1.57-0.18, 0.25+0.06, 1.57,1.57-0.78, 1.57};

        this->declare_parameter<std::string>("model_name", "1.onnx");
        this->declare_parameter<std::string>("motion_name", "motion.npz");
        this->declare_parameter<std::string>("motion_model_name", "1.onnx");
        this->declare_parameter<float>("act_alpha", 0.9);
        this->declare_parameter<float>("gyro_alpha", 0.9);
        this->declare_parameter<float>("angle_alpha", 0.9);
        this->declare_parameter<int>("intra_threads", -1);
        this->declare_parameter<bool>("use_interrupt", false);
        this->declare_parameter<bool>("use_beyondmimic", false);
        this->declare_parameter<bool>("use_attn_enc", false);
        this->declare_parameter<int>("obs_num", 78);
        this->declare_parameter<int>("motion_obs_num", 121);
        this->declare_parameter<int>("perception_obs_num", 187);
        this->declare_parameter<int>("frame_stack", 15);
        this->declare_parameter<int>("motion_frame_stack", 1);
        this->declare_parameter<int>("joint_num", 23);
        this->declare_parameter<int>("decimation", 10);
        this->declare_parameter<float>("dt", 0.001);
        this->declare_parameter<float>("obs_scales_lin_vel", 1.0);
        this->declare_parameter<float>("obs_scales_ang_vel", 1.0);
        this->declare_parameter<float>("obs_scales_dof_pos", 1.0);
        this->declare_parameter<float>("obs_scales_dof_vel", 1.0);
        this->declare_parameter<float>("obs_scales_gravity_b", 1.0);
        this->declare_parameter<float>("clip_observations", 100.0);
        this->declare_parameter<float>("action_scale", 0.3);
        this->declare_parameter<float>("clip_actions", 18.0);
        this->declare_parameter<std::vector<long int>>(
            "usd2urdf", std::vector<long int>{0, 6,  12, 1, 7,  13, 18, 2, 8,  14, 19, 3,
                                              9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22});

        this->get_parameter("model_name", model_name_);
        this->get_parameter("motion_name", motion_name_);
        this->get_parameter("motion_model_name", motion_model_name_);
        this->get_parameter("act_alpha", act_alpha_);
        this->get_parameter("gyro_alpha", gyro_alpha_);
        this->get_parameter("angle_alpha", angle_alpha_);
        this->get_parameter("intra_threads", intra_threads_);
        this->get_parameter("use_interrupt", use_interrupt_);
        this->get_parameter("use_beyondmimic", use_beyondmimic_);
        this->get_parameter("use_attn_enc", use_attn_enc_);
        this->get_parameter("obs_num", obs_num_);
        this->get_parameter("motion_obs_num", motion_obs_num_);
        this->get_parameter("perception_obs_num", perception_obs_num_);
        this->get_parameter("frame_stack", frame_stack_);
        this->get_parameter("motion_frame_stack", motion_frame_stack_);
        this->get_parameter("joint_num", joint_num_);
        this->get_parameter("decimation", decimation_);
        this->get_parameter("dt", dt_);
        this->get_parameter("obs_scales_lin_vel", obs_scales_lin_vel_);
        this->get_parameter("obs_scales_ang_vel", obs_scales_ang_vel_);
        this->get_parameter("obs_scales_dof_pos", obs_scales_dof_pos_);
        this->get_parameter("obs_scales_dof_vel", obs_scales_dof_vel_);
        this->get_parameter("obs_scales_gravity_b", obs_scales_gravity_b_);
        this->get_parameter("clip_observations", clip_observations_);
        this->get_parameter("action_scale", action_scale_);
        this->get_parameter("clip_actions", clip_actions_);
        usd2urdf_.resize(joint_num_);
        this->get_parameter("usd2urdf", usd2urdf_);

        model_path_ = std::string(ROOT_DIR) + "models/" + model_name_;
        motion_path_ = std::string(ROOT_DIR) + "motions/" + motion_name_;
        motion_model_path_ = std::string(ROOT_DIR) + "models/" + motion_model_name_;
        RCLCPP_INFO(this->get_logger(), "model_path: %s", model_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "motion_path: %s", motion_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "motion_model_path: %s", motion_model_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "act_alpha: %f", act_alpha_);
        RCLCPP_INFO(this->get_logger(), "gyro_alpha: %f", gyro_alpha_);
        RCLCPP_INFO(this->get_logger(), "angle_alpha: %f", angle_alpha_);
        RCLCPP_INFO(this->get_logger(), "intra_threads: %d", intra_threads_);
        RCLCPP_INFO(this->get_logger(), "use_interrupt: %s", use_interrupt_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "use_beyondmimic: %s", use_beyondmimic_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "obs_num: %d", obs_num_);
        RCLCPP_INFO(this->get_logger(), "frame_stack: %d", frame_stack_);
        RCLCPP_INFO(this->get_logger(), "joint_num: %d", joint_num_);
        RCLCPP_INFO(this->get_logger(), "decimation: %d", decimation_);
        RCLCPP_INFO(this->get_logger(), "dt: %f", dt_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_lin_vel: %f", obs_scales_lin_vel_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_ang_vel: %f", obs_scales_ang_vel_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_dof_pos: %f", obs_scales_dof_pos_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_dof_vel: %f", obs_scales_dof_vel_);
        RCLCPP_INFO(this->get_logger(), "obs_scales_gravity_b: %f", obs_scales_gravity_b_);
        RCLCPP_INFO(this->get_logger(), "action_scale: %f", action_scale_);
        RCLCPP_INFO(this->get_logger(), "clip_actions: %f", clip_actions_);
        RCLCPP_INFO(this->get_logger(),
                    "usd2urdf: %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, "
                    "%ld, %ld, %ld, %ld",
                    usd2urdf_[0], usd2urdf_[1], usd2urdf_[2], usd2urdf_[3], usd2urdf_[4], usd2urdf_[5],
                    usd2urdf_[6], usd2urdf_[7], usd2urdf_[8], usd2urdf_[9], usd2urdf_[10], usd2urdf_[11],
                    usd2urdf_[12], usd2urdf_[13], usd2urdf_[14], usd2urdf_[15], usd2urdf_[16], usd2urdf_[17],
                    usd2urdf_[18], usd2urdf_[19], usd2urdf_[20], usd2urdf_[21], usd2urdf_[22]);

        Ort::ThreadingOptions thread_opts;
        if (intra_threads_ > 0) {
            thread_opts.SetGlobalIntraOpNumThreads(intra_threads_);
        }
        env_ = std::make_unique<Ort::Env>(thread_opts, ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeInference");
        if(use_attn_enc_){
            setup_model(normal_ctx_, model_path_, obs_num_ * frame_stack_ + perception_obs_num_);
        } else {
            setup_model(normal_ctx_, model_path_, obs_num_ * frame_stack_);
        }
        if(use_beyondmimic_){
             setup_model(motion_ctx_, motion_model_path_, motion_obs_num_ * motion_frame_stack_);
        }
        active_ctx_ = normal_ctx_.get();

        if(use_beyondmimic_){
            try{
                motion_loader_ = std::make_unique<MotionLoader>(motion_path_);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                exit(1);
            }
        }

        obs_ = std::vector<float>(obs_num_, 0.0);
        joint_obs_ = std::vector<float>(joint_num_ * 2, 0.0);
        motion_pos_ = std::vector<float>(joint_num_, 0.0);
        motion_vel_ = std::vector<float>(joint_num_, 0.0);
        act_ = std::make_shared<std::vector<float>>(joint_num_, 0.0);
        tmp_act_ = std::make_shared<std::vector<float>>(joint_num_, 0.0);
        last_act_ = std::vector<float>(joint_num_, 0.0);
        write_buffer_ = std::make_shared<SensorData>(perception_obs_num_);
        write_buffer_->imu_obs[0] = 1.0;
        tmp_data_ = std::make_shared<SensorData>(perception_obs_num_);
        tmp_data_->imu_obs[0] = 1.0;
        if (use_attn_enc_){
            perception_obs_ = std::vector<float>(perception_obs_num_, 0.0);
        }
        reset();

        auto sensor_data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        auto control_command_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
        left_leg_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_left_leg", sensor_data_qos,
            std::bind(&InferenceNode::subs_left_leg_callback, this, std::placeholders::_1));
        right_leg_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_right_leg", sensor_data_qos,
            std::bind(&InferenceNode::subs_right_leg_callback, this, std::placeholders::_1));
        left_arm_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_left_arm", sensor_data_qos,
            std::bind(&InferenceNode::subs_left_arm_callback, this, std::placeholders::_1));
        right_arm_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_right_arm", sensor_data_qos,
            std::bind(&InferenceNode::subs_right_arm_callback, this, std::placeholders::_1));
        IMU_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/IMU_data", sensor_data_qos, std::bind(&InferenceNode::subs_IMU_callback, this, std::placeholders::_1));
        joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", sensor_data_qos, std::bind(&InferenceNode::subs_joy_callback, this, std::placeholders::_1));
        cmd_subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", sensor_data_qos, std::bind(&InferenceNode::subs_cmd_callback,this, std::placeholders::_1
        ));
        elevation_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/robot_0/elevation_data", sensor_data_qos,
            std::bind(&InferenceNode::subs_elevation_callback, this, std::placeholders::_1));
        left_leg_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_left_leg", control_command_qos);
        right_leg_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_right_leg", control_command_qos);
        left_arm_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_left_arm", control_command_qos);
        right_arm_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_command_right_arm", control_command_qos);
        inference_thread_ = std::thread(&InferenceNode::inference, this);
        timer_pub_ = this->create_wall_timer(std::chrono::milliseconds((int)(dt_ * 1000)),
                                             std::bind(&InferenceNode::publish_joint_states, this));
    }
    ~InferenceNode() {
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
    }
    struct SensorData {
        SensorData(int elevation_num=0, int left_leg_num=12, int right_leg_num=14, int left_arm_num=10,
             int right_arm_num=10, int imu_num=7)
            : left_leg_obs(left_leg_num, 0.0),
              right_leg_obs(right_leg_num, 0.0),
              left_arm_obs(left_arm_num, 0.0),
              right_arm_obs(right_arm_num, 0.0),
              imu_obs(imu_num, 0.0),
              elevation_obs(elevation_num, 0.0) {}

        float vx = 0.0, vy = 0.0, dyaw = 0.0;
        std::vector<float> left_leg_obs;
        std::vector<float> right_leg_obs;
        std::vector<float> left_arm_obs;
        std::vector<float> right_arm_obs;
        std::vector<float> imu_obs;
        std::vector<float> elevation_obs;
    };
    struct ModelContext {
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::MemoryInfo> memory_info;
        std::unique_ptr<Ort::Value> input_tensor;
        std::unique_ptr<Ort::Value> output_tensor;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<const char *> input_names_raw;
        std::vector<const char *> output_names_raw;
        std::vector<int64_t> input_shape;
        std::vector<int64_t> output_shape;
        std::vector<float> input_buffer;
        std::vector<float> output_buffer;
        size_t num_inputs;
        size_t num_outputs;
    };
   private:
    std::shared_ptr<SensorData> write_buffer_, tmp_data_;
    std::atomic<bool> is_running_{false}, is_joy_control_{true}, is_interrupt_{false}, is_beyondmimic_{false};
    std::string model_name_, model_path_, motion_name_, motion_path_, motion_model_name_, motion_model_path_;
    bool use_interrupt_, use_beyondmimic_, use_attn_enc_;
    int obs_num_, motion_obs_num_, perception_obs_num_, frame_stack_, motion_frame_stack_, joint_num_;
    int decimation_;
    std::unique_ptr<Ort::Env> env_;
    int intra_threads_;
    Ort::AllocatorWithDefaultOptions allocator_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr left_leg_publisher_, right_leg_publisher_,
        left_arm_publisher_, right_arm_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr left_leg_subscription_,
        right_leg_subscription_, left_arm_subscription_, right_arm_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr  IMU_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr elevation_subscription_;
    rclcpp::TimerBase::SharedPtr timer_pub_;
    std::thread inference_thread_;
    std::vector<float> obs_, last_act_, perception_obs_, motion_pos_, motion_vel_, joint_obs_;
    std::shared_ptr<std::vector<float>> act_, tmp_act_;
    float act_alpha_, gyro_alpha_, angle_alpha_;
    float dt_;
    float obs_scales_lin_vel_, obs_scales_ang_vel_, obs_scales_dof_pos_, obs_scales_dof_vel_,
        obs_scales_gravity_b_, clip_observations_;
    float action_scale_, clip_actions_;
    std::vector<long int> usd2urdf_;
    bool is_first_frame_;
    std::vector<float> joint_limits_lower_, joint_limits_upper_;
    float gravity_z_upper_;
    int last_button0_ = 0, last_button1_ = 0, last_button2_ = 0, last_button3_ = 0, last_button4_ = 0;
    std::shared_ptr<MotionLoader> motion_loader_;
    size_t motion_frame_ = 0;
    std::vector<const char *> input_names_raw_, output_names_raw_;
    std::unique_ptr<ModelContext> normal_ctx_, motion_ctx_;
    ModelContext* active_ctx_;

    void subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg);
    void subs_left_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_right_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_left_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_right_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void subs_IMU_callback(const std::shared_ptr<sensor_msgs::msg::Imu> msg);
    void subs_cmd_callback(const std::shared_ptr<geometry_msgs::msg::Twist> msg);
    void subs_elevation_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg);
    void publish_joint_states();
    void get_gravity_b(const SensorData& data, int offset);
    void inference();
    void reset();
    void setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size);
};