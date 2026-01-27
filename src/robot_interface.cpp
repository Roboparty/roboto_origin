#include "robot_interface.hpp"

RobotInterface::RobotInterface(const std::string& config_file) {
    YAML::Node config = YAML::LoadFile(config_file);

    imu_cfg_ = std::make_shared<IMUCfg>();
    if (config["imu"]) {
        YAML::Node imu_node = config["imu"];
        if (imu_node["imu_id"]) imu_cfg_->imu_id_ = imu_node["imu_id"].as<int>();
        if (imu_node["baudrate"]) imu_cfg_->baudrate_ = imu_node["baudrate"].as<int>();
        if (imu_node["imu_type"]) imu_cfg_->imu_type_ = imu_node["imu_type"].as<std::string>();
        if (imu_node["imu_interface_type"]) imu_cfg_->imu_interface_type_ = imu_node["imu_interface_type"].as<std::string>();
        if (imu_node["imu_interface"]) imu_cfg_->imu_interface_ = imu_node["imu_interface"].as<std::string>();
        setup_imu();
    }

    motors_cfg_ = std::make_shared<MotorsCfg>();
    if (config["motors"]) {
        YAML::Node motors_node = config["motors"];
        if (motors_node["master_id_offset"]) motors_cfg_->master_id_offset_ = motors_node["master_id_offset"].as<int>();
        if (motors_node["motor_type"]) motors_cfg_->motor_type_ = motors_node["motor_type"].as<std::string>();
        if (motors_node["motor_interface_type"]) motors_cfg_->motor_interface_type_ = motors_node["motor_interface_type"].as<std::string>();
        if (motors_node["motor_interface"]) motors_cfg_->motor_interface_ = motors_node["motor_interface"].as<std::vector<std::string>>();
        if (motors_node["motor_id"]) motors_cfg_->motor_id_ = motors_node["motor_id"].as<std::vector<long int>>();
        if (motors_node["motor_model"]) motors_cfg_->motor_model_ = motors_node["motor_model"].as<std::vector<long int>>();
        if (motors_node["motor_num"]) motors_cfg_->motor_num_ = motors_node["motor_num"].as<std::vector<long int>>();
        setup_motors();
    } else {
        throw std::runtime_error("Motors configuration not found in " + config_file);
    }

    robot_cfg_ = std::make_shared<RobotCfg>();
    if (config["robot"]) {
        YAML::Node robot_node = config["robot"];
        if (robot_node["kp"]) robot_cfg_->kp_ = robot_node["kp"].as<std::vector<double>>();
        if (robot_node["kd"]) robot_cfg_->kd_ = robot_node["kd"].as<std::vector<double>>();
        if (robot_node["close_chain_motor_id"]) robot_cfg_->close_chain_motor_id_ = robot_node["close_chain_motor_id"].as<std::vector<long int>>();
        if (robot_node["motor_sign"]) robot_cfg_->motor_sign_ = robot_node["motor_sign"].as<std::vector<long int>>();
    } else {
        throw std::runtime_error("Robot configuration not found in " + config_file);
    }

    thread_pool_ = std::make_unique<ThreadPool>(motors_cfg_->motor_interface_.size());

    ankle_decouple_ = std::make_shared<Decouple>();

    joint_q_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    joint_vel_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
    joint_tau_ = std::vector<float>(motors_cfg_->motor_id_.size(), 0.0);
}

void RobotInterface::setup_motors(){
    size_t count = 0;
    motors_.resize(motors_cfg_->motor_id_.size());
    for (size_t i = 0; i < motors_cfg_->motor_interface_.size(); ++i){
        for (size_t j = 0; j < motors_cfg_->motor_num_[i]; ++j){
            motors_[count] = MotorDriver::create_motor(motors_cfg_->motor_id_[count], motors_cfg_->motor_interface_type_, motors_cfg_->motor_interface_[i], motors_cfg_->motor_type_, motors_cfg_->motor_model_[count], motors_cfg_->master_id_offset_);
            count += 1;
        }
    }
}

void RobotInterface::setup_imu(){
    imu_ = IMUDriver::create_imu(imu_cfg_->imu_id_, imu_cfg_->imu_interface_type_, imu_cfg_->imu_interface_, imu_cfg_->imu_type_, imu_cfg_->baudrate_);
}

void RobotInterface::apply_action(std::vector<float> action) {
    if(!is_init_.load()){
        return;
    }

    {
        std::unique_lock<std::mutex> lock(joint_mutex_);
        std::vector<std::function<void()>> tasks;
        size_t count = 0;
        exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int idx) {
            joint_q_[idx] = motor->get_motor_pos() * robot_cfg_->motor_sign_[idx];
            joint_vel_[idx] = motor->get_motor_spd() * robot_cfg_->motor_sign_[idx];
            joint_tau_[idx] = motor->get_motor_current() * robot_cfg_->motor_sign_[idx];
            if (motor->get_response_count() > offline_threshold_) {
                throw std::runtime_error("Motor " + std::to_string(idx) + " offline");
            }
        });

        if (!robot_cfg_->close_chain_motor_id_.empty()){
            process_close_chain(joint_q_.data(), joint_vel_.data(), joint_tau_.data(), action.data(), false);
        }
    }

    exec_motors_parallel([this, &action](std::shared_ptr<MotorDriver>& motor, int idx) {
        if (std::find(robot_cfg_->close_chain_motor_id_.begin(), robot_cfg_->close_chain_motor_id_.end(), idx) == robot_cfg_->close_chain_motor_id_.end()){
            motor->motor_mit_cmd(action[idx] * robot_cfg_->motor_sign_[idx], 0.0f, robot_cfg_->kp_[idx], robot_cfg_->kd_[idx], 0.0f);
        } else {
            motor->motor_mit_cmd(0.0f, 0.0f, 0.0f, 0.0f, action[idx] * robot_cfg_->motor_sign_[idx]);
        }
    });
}

void RobotInterface::reset_motors(std::vector<double> joint_default_angle) {
    size_t count = 0;
    std::vector<float> motor_default_q(joint_default_angle.begin(), joint_default_angle.end());

    if (!robot_cfg_->close_chain_motor_id_.empty()){
        std::vector<float> dummy_vel(motors_cfg_->motor_id_.size(), 0.0f);
        std::vector<float> dummy_tau(motors_cfg_->motor_id_.size(), 0.0f);
        std::vector<float> dummy_target(motors_cfg_->motor_id_.size(), 0.0f);
        process_close_chain(motor_default_q.data(), dummy_vel.data(), dummy_tau.data(), dummy_target.data(), true);
    }

    exec_motors_parallel([this, &motor_default_q](std::shared_ptr<MotorDriver>& motor, int idx) {
        if (std::find(robot_cfg_->close_chain_motor_id_.begin(), robot_cfg_->close_chain_motor_id_.end(), idx) == robot_cfg_->close_chain_motor_id_.end()) {
                motor->motor_mit_cmd(motor_default_q[idx] * robot_cfg_->motor_sign_[idx], 0.0f, robot_cfg_->kp_[idx]/2.0f, robot_cfg_->kd_[idx]/2.0f, 0.0f);
        } else {
                motor->motor_mit_cmd(motor_default_q[idx] * robot_cfg_->motor_sign_[idx], 0.0f, 0.0f, 0.0f, 0.0f);
        }
    });
}

void RobotInterface::read_motors() {
    {
        std::unique_lock<std::mutex> lock(joint_mutex_);
        exec_motors_parallel([this](std::shared_ptr<MotorDriver>& motor, int idx) {
            motor->refresh_motor_status();
            joint_q_[idx] = motor->get_motor_pos() * robot_cfg_->motor_sign_[idx];
            joint_vel_[idx] = motor->get_motor_spd() * robot_cfg_->motor_sign_[idx];
            joint_tau_[idx] = motor->get_motor_current() * robot_cfg_->motor_sign_[idx];
        });

        if (!robot_cfg_->close_chain_motor_id_.empty()) {
             std::vector<float> dummy_target(motors_cfg_->motor_id_.size(), 0.0f);
             process_close_chain(joint_q_.data(), joint_vel_.data(), joint_tau_.data(), dummy_target.data(), true);
        }
    }
}

void RobotInterface::set_zeros() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->set_motor_zero();
    });
}

void RobotInterface::clear_errors() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->clear_motor_error();
    });
}

void RobotInterface::init_motors() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->init_motor();
    });
    is_init_.store(true);
}

void RobotInterface::deinit_motors() {
    exec_motors_parallel([](std::shared_ptr<MotorDriver>& motor, int idx) {
        motor->deinit_motor();
    });
    is_init_.store(false);
}

void RobotInterface::exec_motors_parallel(std::function<void(std::shared_ptr<MotorDriver>&, int)> cmd_func) {
    std::unique_lock<std::mutex> lock(motors_mutex_);
    std::vector<std::function<void()>> tasks;
    size_t count = 0;
    
    for (size_t i = 0; i < motors_cfg_->motor_interface_.size(); ++i) {
        size_t num_motors = motors_cfg_->motor_num_[i];
        size_t start_idx = count;
        tasks.push_back([this, start_idx, num_motors, cmd_func]() {
            for (size_t j = 0; j < num_motors; ++j) {
                size_t idx = start_idx + j;
                cmd_func(motors_[idx], idx); 
            }
        });
        count += num_motors;
    }
    thread_pool_->run_parallel(tasks);
}

void RobotInterface::process_close_chain(float* q_in, float* vel_in, float* tau_in, float* target, bool forward_only) {
    Eigen::VectorXd q(2), vel(2), tau(2);
    int idx1 = robot_cfg_->close_chain_motor_id_[0];
    int idx2 = robot_cfg_->close_chain_motor_id_[1];

    q << q_in[idx1], q_in[idx2];
    vel << vel_in[idx1], vel_in[idx2];
    tau << tau_in[idx1], tau_in[idx2];

    ankle_decouple_->get_forwardQVT(q, vel, tau, true);
    q_in[idx1] = q[0];
    q_in[idx2] = q[1];
    vel_in[idx1] = vel[0];
    vel_in[idx2] = vel[1];
    tau_in[idx1] = tau[0];
    tau_in[idx2] = tau[1];

    if (!forward_only) {
        tau << robot_cfg_->kp_[idx1] * (q[0] - target[idx1]) + robot_cfg_->kd_[idx1] * (0.0f - vel[0]),
               robot_cfg_->kp_[idx2] * (q[1] - target[idx2]) + robot_cfg_->kd_[idx2] * (0.0f - vel[1]);
        ankle_decouple_->get_decoupleQVT(q, vel, tau, true);
        target[idx1] = tau[0];
        target[idx2] = tau[1];
    }

    idx1 = robot_cfg_->close_chain_motor_id_[2];
    idx2 = robot_cfg_->close_chain_motor_id_[3];
    q << q_in[idx1], q_in[idx2];
    vel << vel_in[idx1], vel_in[idx2];
    tau << tau_in[idx1], tau_in[idx2];

    ankle_decouple_->get_forwardQVT(q, vel, tau, false);
    q_in[idx1] = q[0];
    q_in[idx2] = q[1];
    vel_in[idx1] = vel[0];
    vel_in[idx2] = vel[1];
    tau_in[idx1] = tau[0];
    tau_in[idx2] = tau[1];

    if (!forward_only) {
        tau << robot_cfg_->kp_[idx1] * (q[0] - target[idx1]) + robot_cfg_->kd_[idx1] * (0.0f - vel[0]),
               robot_cfg_->kp_[idx2] * (q[1] - target[idx2]) + robot_cfg_->kd_[idx2] * (0.0f - vel[1]);
        ankle_decouple_->get_decoupleQVT(q, vel, tau, false);
        target[idx1] = tau[0];
        target[idx2] = tau[1];
    }
}
