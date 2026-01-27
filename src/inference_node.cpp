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
    std::fill(joint_pos_.begin(), joint_pos_.end(), 0.0f);
    std::fill(joint_vel_.begin(), joint_vel_.end(), 0.0f);
    std::fill(motion_pos_.begin(), motion_pos_.end(), 0.0f);
    std::fill(motion_vel_.begin(), motion_vel_.end(), 0.0f);
    std::fill(cmd_vel_.begin(), cmd_vel_.end(), 0.0f);
    std::fill(quat_.begin(), quat_.end(), 0.0f);
    std::fill(ang_vel_.begin(), ang_vel_.end(), 0.0f);
    if (active_ctx_) {
        std::fill(active_ctx_->input_buffer.begin(), active_ctx_->input_buffer.end(), 0.0f);
        std::fill(active_ctx_->output_buffer.begin(), active_ctx_->output_buffer.end(), 0.0f);
    }
    std::fill(act_.begin(), act_.end(), 0.0f);
    std::fill(last_act_.begin(), last_act_.end(), 0.0f);
    std::fill(joint_torques_.begin(), joint_torques_.end(), 0.0f);
    is_first_frame_ = true;
    motion_frame_ = 0;
    is_interrupt_.store(false);
    is_beyondmimic_.store(false);
    if(use_interrupt_){
        std::fill(interrupt_action_.begin(), interrupt_action_.end(), 0.0f);
    }
    if(use_attn_enc_){
        std::fill(perception_obs_.begin(), perception_obs_.end(), 0.0f);
    }
}

void InferenceNode::apply_action() {
    if(!is_running_.load() || !robot_->is_init_.load()){
        return;
    }
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        for (size_t i = 0; i < act_.size(); i++) {
            act_[i] = act_alpha_ * act_[i] + (1 - act_alpha_) * last_act_[i];
        }
        last_act_ = act_;
        robot_->apply_action(act_);
    }
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

        ang_vel_ = robot_->get_ang_vel();
        for (int i = 0; i < 3; i++) {
            obs_[i + offset] = ang_vel_[i] * obs_scales_ang_vel_;
        }
        offset += 3;
        quat_ = robot_->get_quat();
        Eigen::Quaternionf q_b2w(quat_[0], quat_[1], quat_[2], quat_[3]);
        Eigen::Vector3f gravity_w(0.0f, 0.0f, -1.0f);
        Eigen::Quaternionf q_w2b = q_b2w.inverse();
        Eigen::Vector3f gravity_b = q_w2b * gravity_w;
        if (gravity_b.z() > gravity_z_upper_){
            RCLCPP_FATAL(this->get_logger(), "Robot fell down! Shutting down...");
            rclcpp::shutdown();
            return;
        }
        obs_[0 + offset] = gravity_b.x() * obs_scales_gravity_b_;
        obs_[1 + offset] = gravity_b.y() * obs_scales_gravity_b_;
        obs_[2 + offset] = gravity_b.z() * obs_scales_gravity_b_;
        offset += 3;
        publish_imu();

        if (!is_beyondmimic_.load()){
            std::unique_lock<std::mutex> lock(cmd_mutex_);
            obs_[0 + offset] = cmd_vel_[0] * obs_scales_lin_vel_;
            obs_[1 + offset] = cmd_vel_[1] * obs_scales_lin_vel_;
            obs_[2 + offset] = cmd_vel_[2] * obs_scales_ang_vel_;
            offset += 3;
        }

        joint_pos_ = robot_->get_joint_q();
        joint_vel_ = robot_->get_joint_vel();
        joint_torques_ = robot_->get_joint_tau();
        for (int i = 0; i < joint_num_; i++) {
            obs_[offset + i] = joint_pos_[usd2urdf_[i]] * obs_scales_dof_pos_;
            obs_[offset + joint_num_ + i] = joint_vel_[usd2urdf_[i]] * obs_scales_dof_vel_;
        }
        for(size_t i = 0; i < joint_limits_.size() / 2; i++){
            if(joint_pos_[i] < joint_limits_[i * 2] || joint_pos_[i] > joint_limits_[i * 2 + 1]){
                RCLCPP_FATAL(this->get_logger(), "Joint %ld out of limit! Shutting down...", i);
                rclcpp::shutdown();
                return;
            }
        }
        publish_joint_states();
        offset += joint_num_ * 2;

        for (int i = 0; i < joint_num_; i++) {
            obs_[offset + i] = active_ctx_->output_buffer[i];
        }
        offset += joint_num_;

        if (use_interrupt_){
            obs_[offset] = is_interrupt_.load() ? 1.0 : 0.0;
            offset += 1;
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
                std::unique_lock<std::mutex> lock(perception_mutex_);
                std::copy(perception_obs_.begin(), perception_obs_.end(), active_ctx_->input_buffer.begin() + frame_stack * obs_num);
            }
            is_first_frame_ = false;
        } else {
            std::copy(active_ctx_->input_buffer.begin() + obs_num, active_ctx_->input_buffer.begin() + frame_stack * obs_num, active_ctx_->input_buffer.begin());
            std::copy(obs_.begin(), obs_.end(), active_ctx_->input_buffer.begin() + (frame_stack - 1) * obs_num);
            if(use_attn_enc_){
                std::unique_lock<std::mutex> lock(perception_mutex_);
                std::copy(perception_obs_.begin(), perception_obs_.end(), active_ctx_->input_buffer.begin() + frame_stack * obs_num);
            }
        }

        active_ctx_->session->Run(Ort::RunOptions{nullptr}, 
            active_ctx_->input_names_raw.data(), active_ctx_->input_tensor.get(), active_ctx_->num_inputs,
            active_ctx_->output_names_raw.data(), active_ctx_->output_tensor.get(), active_ctx_->num_outputs);
        
        {
            std::unique_lock<std::mutex> lock(act_mutex_);
            for (int i = 0; i < active_ctx_->output_buffer.size(); i++) {
                active_ctx_->output_buffer[i] = std::clamp(active_ctx_->output_buffer[i], -clip_actions_, clip_actions_);
                act_[usd2urdf_[i]] = active_ctx_->output_buffer[i];
                act_[usd2urdf_[i]] = act_[usd2urdf_[i]] * action_scale_ + joint_default_angle_[usd2urdf_[i]];
            }
            if(use_interrupt_ && is_interrupt_.load()){
                std::unique_lock<std::mutex> lock(interrupt_mutex_);
                for (size_t i = 0; i < 10; i++) {
                    act_[14 + i] = interrupt_action_[i];
                }
            }
            publish_action();
        }

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
    try {
        auto node = std::make_shared<InferenceNode>();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
        executor.add_node(node);
        RCLCPP_INFO(node->get_logger(), "Press 'A' to initialize/deinitialize motors");
        RCLCPP_INFO(node->get_logger(), "Press 'X' to reset motors");
        RCLCPP_INFO(node->get_logger(), "Press 'B' to start/pause inference");
        RCLCPP_INFO(node->get_logger(), "Press 'Y' to switch between joystick and /cmd_vel control");
        executor.spin();
    } catch (const std::exception &e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Exception caught: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
