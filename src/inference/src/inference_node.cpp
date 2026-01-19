#include "inference_node.hpp"

void InferenceNode::setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size){
    if (!ctx) {
        ctx = std::make_unique<ModelContext>();
    }

    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    ctx->session = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    
    ctx->num_inputs = ctx->session->GetInputCount();
    ctx->input_names.resize(ctx->num_inputs);
    ctx->input_buffer.resize(input_size);

    for (size_t i = 0; i < ctx->num_inputs; i++) {
        Ort::AllocatedStringPtr input_name = ctx->session->GetInputNameAllocated(i, allocator_);
        ctx->input_names[i] = input_name.get();
        auto type_info = ctx->session->GetInputTypeInfo(i);
        ctx->input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (ctx->input_shape[0] == -1) ctx->input_shape[0] = 1;
    }

    ctx->num_outputs = ctx->session->GetOutputCount();
    ctx->output_names.resize(ctx->num_outputs);
    ctx->output_buffer.resize(joint_num_);

    for (size_t i = 0; i < ctx->num_outputs; i++) {
        Ort::AllocatedStringPtr output_name = ctx->session->GetOutputNameAllocated(i, allocator_);
        ctx->output_names[i] = output_name.get();
        auto type_info = ctx->session->GetOutputTypeInfo(i);
        ctx->output_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    }

    ctx->input_names_raw = std::vector<const char *>(ctx->num_inputs, nullptr);
    ctx->output_names_raw = std::vector<const char *>(ctx->num_outputs, nullptr);
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        ctx->input_names_raw[i] = ctx->input_names[i].c_str();
    }
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        ctx->output_names_raw[i] = ctx->output_names[i].c_str();
    }

    ctx->memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    
    ctx->input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->input_buffer.data(), ctx->input_buffer.size(), ctx->input_shape.data(), ctx->input_shape.size()));
        
    ctx->output_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->output_buffer.data(), ctx->output_buffer.size(), ctx->output_shape.data(), ctx->output_shape.size()));
}

void InferenceNode::reset() {
    is_running_.store(false);
    std::fill(obs_.begin(), obs_.end(), 0.0f);
    std::fill(joint_obs_.begin(), joint_obs_.end(), 0.0f);
    std::fill(motion_pos_.begin(), motion_pos_.end(), 0.0f);
    std::fill(motion_vel_.begin(), motion_vel_.end(), 0.0f);
    if (active_ctx_) {
        std::fill(active_ctx_->input_buffer.begin(), active_ctx_->input_buffer.end(), 0.0f);
        std::fill(active_ctx_->output_buffer.begin(), active_ctx_->output_buffer.end(), 0.0f);
    }
    std::fill(last_act_.begin(), last_act_.end(), 0.0f);
    auto new_act = std::make_shared<std::vector<float>>(joint_num_, 0.0);
    auto new_tmp_act = std::make_shared<std::vector<float>>(joint_num_, 0.0);
    std::atomic_store(&act_, new_act);
    std::atomic_store(&tmp_act_, new_tmp_act);
    auto new_write_buffer = std::make_shared<SensorData>(perception_obs_num_);
    new_write_buffer->imu_obs[0] = 1.0;
    std::atomic_store(&write_buffer_, new_write_buffer);
    auto new_tmp_data = std::make_shared<SensorData>(perception_obs_num_);
    new_tmp_data->imu_obs[0] = 1.0;
    std::atomic_store(&tmp_data_, new_tmp_data);
    is_first_frame_ = true;
    motion_frame_ = 0;
    is_interrupt_.store(false);
    is_beyondmimic_.store(false);
    if(use_attn_enc_){
        std::fill(perception_obs_.begin(), perception_obs_.end(), 0.0f);
    }
}

void InferenceNode::subs_cmd_callback(const std::shared_ptr<geometry_msgs::msg::Twist> msg){
    if(!is_joy_control_){
        auto data = std::atomic_load(&write_buffer_); // 原子加载
        data->vx = std::clamp(msg->linear.x, -0.4, 0.6);
        data->vy = std::clamp(msg->linear.y, -0.4, 0.4);
        data->dyaw = std::clamp(msg->angular.z, -1.0, 1.0);
    }
}

void InferenceNode::subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg) {
    if (is_joy_control_){
        auto data = std::atomic_load(&write_buffer_); // 原子加载
        data->vx = std::clamp(msg->axes[3] * 0.6, -0.4, 0.6);
        data->vy = std::clamp(msg->axes[2] * 0.4, -0.4, 0.4);
            if (msg->buttons[6] == 1) {
            data->dyaw = std::clamp(msg->buttons[6] * 1.0, 0.0, 1.0);
            } else if (msg->buttons[7] == 1) {
            data->dyaw = std::clamp(-msg->buttons[7] * 1.0, -1.0, 0.0);
            } else {
            data->dyaw = 0.0;
        }
    }
    if ((msg->buttons[0] == 1 && msg->buttons[0] != last_button0_) || (msg->buttons[1] == 1 && msg->buttons[1] != last_button1_)) {
        reset();
        RCLCPP_INFO(this->get_logger(), "Inference paused");
    }
    if (msg->buttons[2] == 1 && msg->buttons[2] != last_button2_) {
        is_running_.store(!is_running_.load());
        RCLCPP_INFO(this->get_logger(), "Inference %s", is_running_.load() ? "started" : "paused");
    }
    if (msg->buttons[3] == 1 && msg->buttons[3] != last_button3_) {
        is_joy_control_.store(!is_joy_control_);
        RCLCPP_INFO(this->get_logger(), "Controlled by %s", is_joy_control_.load() ? "joy" : "/cmd_vel");
    }
    if (use_interrupt_ || use_beyondmimic_) {
        if (msg->buttons[4] == 1 && msg->buttons[4] != last_button4_) {
            if(use_interrupt_){
                is_interrupt_.store(!is_interrupt_.load());
                RCLCPP_INFO(this->get_logger(), "Interrupt mode %s", is_interrupt_.load() ? "enabled" : "disabled");
            }
            if(use_beyondmimic_){
                is_running_.store(false);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                is_beyondmimic_.store(!is_beyondmimic_.load());
                bool is_beyondmimic = is_beyondmimic_.load();
                active_ctx_ = is_beyondmimic ? motion_ctx_.get() : normal_ctx_.get();
                int obs_num = is_beyondmimic ? motion_obs_num_ : obs_num_;
                obs_.resize(obs_num);
                std::fill(obs_.begin(), obs_.end(), 0.0f);
                std::fill(joint_obs_.begin(), joint_obs_.end(), 0.0f);
                std::fill(motion_pos_.begin(), motion_pos_.end(), 0.0f);
                std::fill(motion_vel_.begin(), motion_vel_.end(), 0.0f);
                std::fill(active_ctx_->input_buffer.begin(), active_ctx_->input_buffer.end(), 0.0f);
                std::fill(active_ctx_->output_buffer.begin(), active_ctx_->output_buffer.end(), 0.0f);
                std::fill(last_act_.begin(), last_act_.end(), 0.0f);
                is_first_frame_ = true;
                motion_frame_ = 0;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                is_running_.store(true);
                RCLCPP_INFO(this->get_logger(), "Beyondmimic mode %s", is_beyondmimic ? "enabled" : "disabled");
            }
        }
        last_button4_ = msg->buttons[4];
    }
    last_button0_ = msg->buttons[0];
    last_button1_ = msg->buttons[1];
    last_button2_ = msg->buttons[2];
    last_button3_ = msg->buttons[3];
}

void InferenceNode::subs_left_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    for (int i = 0; i < 6; i++) {
        if (msg->position[i] > joint_limits_upper_[i] || msg->position[i] < joint_limits_lower_[i]){
            is_running_.store(false);
            RCLCPP_WARN(this->get_logger(), "Left leg joint %d out of limits, inference paused!", i+1);
            return;
        }
        data->left_leg_obs[i] = msg->position[i];
        data->left_leg_obs[6 + i] = msg->velocity[i];
    }
}

void InferenceNode::subs_right_leg_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    for (int i = 0; i < 7; i++) {
        if (msg->position[i] > joint_limits_upper_[6+i] || msg->position[i] < joint_limits_lower_[6+i]){
            is_running_.store(false);
            RCLCPP_WARN(this->get_logger(), "Right leg joint %d out of limits, inference paused!", i+1);
            return;
        }
        data->right_leg_obs[i] = msg->position[i];
        data->right_leg_obs[7 + i] = msg->velocity[i];
    }
}

void InferenceNode::subs_left_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    for (int i = 0; i < 5; i++) {
        if (msg->position[i] > joint_limits_upper_[13+i] || msg->position[i] < joint_limits_lower_[13+i]){
            is_running_.store(false);
            RCLCPP_WARN(this->get_logger(), "Left arm joint %d out of limits, inference paused!", i+1);
            return;
        }
        data->left_arm_obs[i] = msg->position[i];
        data->left_arm_obs[5 + i] = msg->velocity[i];
    }
}

void InferenceNode::subs_right_arm_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    for (int i = 0; i < 5; i++) {
        if (msg->position[i] > joint_limits_upper_[18+i] || msg->position[i] < joint_limits_lower_[18+i]){
            is_running_.store(false);
            RCLCPP_WARN(this->get_logger(), "Right arm joint %d out of limits, inference paused!", i+1);
            return;
        }
        data->right_arm_obs[i] = msg->position[i];
        data->right_arm_obs[5 + i] = msg->velocity[i];
    }
}

void InferenceNode::subs_IMU_callback(const std::shared_ptr<sensor_msgs::msg::Imu> msg) {
    auto data = std::atomic_load(&write_buffer_); // 原子加载
    data->imu_obs[0] = msg->orientation.w;
    data->imu_obs[1] = msg->orientation.x;
    data->imu_obs[2] = msg->orientation.y;
    data->imu_obs[3] = msg->orientation.z;
    data->imu_obs[4] = gyro_alpha_ * msg->angular_velocity.x + (1 - gyro_alpha_) * data->imu_obs[4];
    data->imu_obs[5] = gyro_alpha_ * msg->angular_velocity.y + (1 - gyro_alpha_) * data->imu_obs[5];
    data->imu_obs[6] = gyro_alpha_ * msg->angular_velocity.z + (1 - gyro_alpha_) * data->imu_obs[6];
}

void InferenceNode::subs_elevation_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg){
    if(use_attn_enc_){
        auto data = std::atomic_load(&write_buffer_); // 原子加载
        std::vector<float>& elevation_obs = data->elevation_obs;
        for(int i = 0; i < perception_obs_num_; i++){
            elevation_obs[i] = msg->data[i];
        }
    }
}

void InferenceNode::publish_joint_states() {
    if(!is_running_.load()){
        return;
    }
    auto act = std::atomic_load(&act_); // 原子加载
    for (size_t i = 0; i < act->size(); i++) {
        (*act)[i] = act_alpha_ * (*act)[i] + (1 - act_alpha_) * last_act_[i];
    }
    last_act_ = *act;
    auto left_leg_message = sensor_msgs::msg::JointState();
    left_leg_message.header.stamp = this->now();
    left_leg_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
    left_leg_message.position = {(*act)[0], (*act)[1], (*act)[2], (*act)[3], (*act)[4], (*act)[5]};
    left_leg_message.velocity = {0, 0, 0, 0, 0, 0};
    left_leg_message.effort = {0, 0, 0, 0, 0, 0};
    left_leg_publisher_->publish(left_leg_message);

    auto right_leg_message = sensor_msgs::msg::JointState();
    right_leg_message.header.stamp = this->now();
    right_leg_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};
    right_leg_message.position = {(*act)[6], (*act)[7], (*act)[8], (*act)[9], (*act)[10], (*act)[11], (*act)[12]};
    right_leg_message.velocity = {0, 0, 0, 0, 0, 0, 0};
    right_leg_message.effort = {0, 0, 0, 0, 0, 0, 0};
    right_leg_publisher_->publish(right_leg_message);

    if(!is_interrupt_.load()){
        auto left_arm_message = sensor_msgs::msg::JointState();
        left_arm_message.header.stamp = this->now();
        left_arm_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5"};
        left_arm_message.position = {(*act)[13], (*act)[14], (*act)[15], (*act)[16], (*act)[17]};
        left_arm_message.velocity = {0, 0, 0, 0, 0};
        left_arm_message.effort = {0, 0, 0, 0, 0};
        left_arm_publisher_->publish(left_arm_message);

        auto right_arm_message = sensor_msgs::msg::JointState();
        right_arm_message.header.stamp = this->now();
        right_arm_message.name = {"joint1", "joint2", "joint3", "joint4", "joint5"};
        right_arm_message.position = {(*act)[18], (*act)[19], (*act)[20], (*act)[21], (*act)[22]};
        right_arm_message.velocity = {0, 0, 0, 0, 0};
        right_arm_message.effort = {0, 0, 0, 0, 0};
        right_arm_publisher_->publish(right_arm_message);
    }
}

void InferenceNode::get_gravity_b(const SensorData& data, int offset) {
    float w, x, y, z;
    w = data.imu_obs[0];
    x = data.imu_obs[1];
    y = data.imu_obs[2];
    z = data.imu_obs[3];

    Eigen::Quaternionf q_b2w(w, x, y, z);
    Eigen::Vector3f gravity_w(0.0f, 0.0f, -1.0f);
    Eigen::Quaternionf q_w2b = q_b2w.inverse();
    Eigen::Vector3f gravity_b = q_w2b * gravity_w;
    if (gravity_b.z() > gravity_z_upper_){
        is_running_.store(false);
        RCLCPP_WARN(this->get_logger(), "Robot fell down! Inference paused.");
        return;
    }

    obs_[0 + offset] = gravity_b.x() * obs_scales_gravity_b_;
    obs_[1 + offset] = gravity_b.y() * obs_scales_gravity_b_;
    obs_[2 + offset] = gravity_b.z() * obs_scales_gravity_b_;
}

void InferenceNode::inference() {
    pthread_setname_np(pthread_self(), "inference");
    struct sched_param sp{}; sp.sched_priority = 65;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000 * 1000 * decimation_));

    while(rclcpp::ok()){
        auto loop_start = std::chrono::steady_clock::now();
        if(!is_running_.load()){
            std::this_thread::sleep_for(period);
            continue;
        }

        tmp_data_ = std::atomic_exchange(&write_buffer_, tmp_data_);
        
        int offset = 0;

        if(is_beyondmimic_.load()){
            motion_pos_ = motion_loader_->get_pos(motion_frame_);
            motion_vel_ = motion_loader_->get_vel(motion_frame_);
            motion_frame_ += 1;
            if(motion_frame_ >= motion_loader_->get_num_frames()){
                motion_frame_ = motion_loader_->get_num_frames() - 1;
            }
            for(int i = 0; i < joint_num_; i++){
                obs_[i + offset] = motion_pos_[i];
                obs_[i + joint_num_ + offset] = motion_vel_[i];
            }
            offset += joint_num_ * 2;
        }

        for (int i = 0; i < 3; i++) {
            obs_[i + offset] = tmp_data_->imu_obs[4 + i] * obs_scales_ang_vel_;
        }
        offset += 3;

        get_gravity_b(*tmp_data_, offset);
        offset += 3;

        if (!is_beyondmimic_.load()){
        obs_[0 + offset] = tmp_data_->vx * obs_scales_lin_vel_;
        obs_[1 + offset] = tmp_data_->vy * obs_scales_lin_vel_;
        obs_[2 + offset] = tmp_data_->dyaw * obs_scales_ang_vel_;
        offset += 3;
        }

        for (int i = 0; i < 6; i++) {
            joint_obs_[i] = tmp_data_->left_leg_obs[i] * obs_scales_dof_pos_;
            joint_obs_[joint_num_ + i] = tmp_data_->left_leg_obs[6 + i] * obs_scales_dof_vel_;
        }
        for (int i = 0; i < 7; i++) {
            joint_obs_[6 + i] = tmp_data_->right_leg_obs[i] * obs_scales_dof_pos_;
            joint_obs_[joint_num_ + 6 + i] = tmp_data_->right_leg_obs[7 + i] * obs_scales_dof_vel_;
        }
        for (int i = 0; i < 5; i++) {
            joint_obs_[13 + i] = tmp_data_->left_arm_obs[i] * obs_scales_dof_pos_;
            joint_obs_[joint_num_ + 13 + i] = tmp_data_->left_arm_obs[5 + i] * obs_scales_dof_vel_;
        }
        for (int i = 0; i < 5; i++) {
            joint_obs_[18 + i] = tmp_data_->right_arm_obs[i] * obs_scales_dof_pos_;
            joint_obs_[joint_num_ + 18 + i] = tmp_data_->right_arm_obs[5 + i] * obs_scales_dof_vel_;
        }
        for (int i = 0; i < joint_num_; i++) {
            obs_[offset + i] = joint_obs_[usd2urdf_[i]];
            obs_[offset + joint_num_ + i] = joint_obs_[joint_num_ + usd2urdf_[i]];
        }
        offset += joint_num_ * 2;

        for (int i = 0; i < joint_num_; i++) {
            obs_[offset + i] = active_ctx_->output_buffer[i];
        }
        offset += joint_num_;

        if (use_interrupt_){
            obs_[offset] = is_interrupt_.load() ? 1.0 : 0.0;
        }

        if (use_attn_enc_){
            for(int i = 0; i < perception_obs_num_; i++){
                perception_obs_[i] = tmp_data_->elevation_obs[i];
            }
        }

        std::transform(obs_.begin(), obs_.end(), obs_.begin(), [this](float val) {
            return std::clamp(val, -clip_observations_, clip_observations_);
        });

        bool is_beyondmimic = is_beyondmimic_.load();
        int obs_num = is_beyondmimic ? motion_obs_num_: obs_num_;
        int frame_stack = is_beyondmimic ? motion_frame_stack_ : frame_stack_;
        if (is_first_frame_) {
            for (int i = 0; i < frame_stack; i++) {
                std::copy(obs_.begin(), obs_.end(), active_ctx_->input_buffer.begin() + i * obs_num);
            }
            if(use_attn_enc_){
                std::copy(perception_obs_.begin(), perception_obs_.end(), active_ctx_->input_buffer.begin() + frame_stack * obs_num);
            }
            is_first_frame_ = false;
        } else {
            std::copy(active_ctx_->input_buffer.begin() + obs_num, active_ctx_->input_buffer.begin() + frame_stack * obs_num, active_ctx_->input_buffer.begin());
            std::copy(obs_.begin(), obs_.end(), active_ctx_->input_buffer.begin() + (frame_stack - 1) * obs_num);
            if(use_attn_enc_){
                std::copy(perception_obs_.begin(), perception_obs_.end(), active_ctx_->input_buffer.begin() + frame_stack * obs_num);
            }
        }

        active_ctx_->session->Run(Ort::RunOptions{nullptr}, 
            active_ctx_->input_names_raw.data(), active_ctx_->input_tensor.get(), active_ctx_->num_inputs,
            active_ctx_->output_names_raw.data(), active_ctx_->output_tensor.get(), active_ctx_->num_outputs);

        for (int i = 0; i < active_ctx_->output_buffer.size(); i++) {
            active_ctx_->output_buffer[i] = std::clamp(active_ctx_->output_buffer[i], -clip_actions_, clip_actions_);
            (*tmp_act_)[usd2urdf_[i]] = active_ctx_->output_buffer[i];
            (*tmp_act_)[usd2urdf_[i]] = (*tmp_act_)[usd2urdf_[i]] * action_scale_;
        }
        tmp_act_ = std::atomic_exchange(&act_, tmp_act_);


        auto loop_end = std::chrono::steady_clock::now();
        // 使用微秒进行计算
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_time = period - elapsed_time;

        if (sleep_time > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        } else {
            // 警告信息也使用更精确的单位
            RCLCPP_WARN(this->get_logger(), "Inference loop overran! Took %ld us, but period is %ld us.", elapsed_time.count(), period.count());
        }
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InferenceNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    RCLCPP_INFO(node->get_logger(), "Press 'B' to start/pause inference");
    executor.spin();
    rclcpp::shutdown();
    return 0;
}