// DmMotorDriver.cpp
#include "dm_motor_driver.hpp"

DM_Limit_Param limit_param[Num_Of_Motor] = {
    {12.5, 20, 28, 500, 5},   // DM4340P_48V
    {12.5, 25, 200, 500, 5},  // DM10010L_48V
};

DmMotorDriver::DmMotorDriver(uint16_t motor_id, const std::string& interface_type, const std::string& can_interface, uint16_t master_id_offset,
                             DM_Motor_Model motor_model)
    : MotorDriver(), can_(SocketCAN::get(can_interface)), motor_model_(motor_model) {
    if (interface_type != "can") {
        throw std::runtime_error("DM driver only support CAN interface");
    }
    motor_id_ = motor_id;
    master_id_ = motor_id_ + master_id_offset;
    limit_param_ = limit_param[motor_model_];
    can_interface_ = can_interface;
    CanCbkFunc can_callback = std::bind(&DmMotorDriver::can_rx_cbk, this, std::placeholders::_1);
    can_->add_can_callback(can_callback, master_id_);
}

DmMotorDriver::~DmMotorDriver() { can_->remove_can_callback(master_id_); }

void DmMotorDriver::lock_motor() {
    can_frame tx_frame;
    tx_frame.can_id = motor_id_;  // change according to the mode
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = 0xFF;
    tx_frame.data[1] = 0xFF;
    tx_frame.data[2] = 0xFF;
    tx_frame.data[3] = 0xFF;
    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    tx_frame.data[7] = 0xFC;

    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

void DmMotorDriver::unlock_motor() {
    can_frame tx_frame;
    tx_frame.can_id = motor_id_;  // change according to the mode
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = 0xFF;
    tx_frame.data[1] = 0xFF;
    tx_frame.data[2] = 0xFF;
    tx_frame.data[3] = 0xFF;
    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    tx_frame.data[7] = 0xFD;
    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

uint8_t DmMotorDriver::init_motor() {
    // send disable command to enter read mode
    DmMotorDriver::unlock_motor();
    Timer::sleep_for(normal_sleep_time);
    set_motor_control_mode(MIT);
    Timer::sleep_for(normal_sleep_time);
    // send enable command to enter contorl mode
    DmMotorDriver::lock_motor();
    Timer::sleep_for(normal_sleep_time);
    DmMotorDriver::refresh_motor_status();
    Timer::sleep_for(normal_sleep_time);
    switch (error_id_) {
        case DMError::DM_DOWN:
            return DMError::DM_DOWN;
            break;
        case DMError::DM_UP:
            return DMError::DM_UP;
            break;
        case DMError::LOST_CONN:
            return DMError::LOST_CONN;
            break;
        case DMError::OVER_CURRENT:
            return DMError::OVER_CURRENT;
            break;
        case DMError::MOS_OVER_TEMP:
            return DMError::MOS_OVER_TEMP;
            break;
        case DMError::COIL_OVER_TEMP:
            return DMError::COIL_OVER_TEMP;
            break;
        case DMError::UNDER_VOLT:
            return DMError::UNDER_VOLT;
            break;
        case DMError::OVER_VOLT:
            return DMError::OVER_VOLT;
            break;
        case DMError::OVER_LOAD:
            return DMError::OVER_LOAD;
            break;
        default:
            return error_id_;
    }
    return error_id_;
}

void DmMotorDriver::deinit_motor() {
    DmMotorDriver::unlock_motor();
    Timer::sleep_for(normal_sleep_time);
}

bool DmMotorDriver::write_motor_flash() { return true; }

bool DmMotorDriver::set_motor_zero() {
    // send set zero command
    DmMotorDriver::set_motor_zero_dm();
    Timer::sleep_for(setup_sleep_time);  // wait for motor to set zero
    logger_->info("motor_id: {0}\tposition: {1}\t", motor_id_, get_motor_pos());
    DmMotorDriver::unlock_motor();
    if (get_motor_pos() > judgment_accuracy_threshold || get_motor_pos() < -judgment_accuracy_threshold) {
        logger_->warn("set zero error");
        return false;
    } else {
        logger_->info("set zero success");
        return false;
    }
    // disable motor
}

void DmMotorDriver::can_rx_cbk(const can_frame& rx_frame) {
    {
        response_count_ = 0;
    }
    uint16_t master_id_t = 0;
    uint16_t pos_int = 0;
    uint16_t spd_int = 0;
    uint16_t t_int = 0;
    pos_int = rx_frame.data[1] << 8 | rx_frame.data[2];
    spd_int = rx_frame.data[3] << 4 | (rx_frame.data[4] & 0xF0) >> 4;
    t_int = (rx_frame.data[4] & 0x0F) << 8 | rx_frame.data[5];
    master_id_t = rx_frame.can_id;
    if ((rx_frame.data[0] & 0xF0) >> 4 > 7) {  // error code range from 8 to 15
        error_id_ = (rx_frame.data[0] & 0xF0) >> 4;
        if (logger_) {
            logger_->error("can_interface: {0}\tmotor_id: {1}\terror_id: 0x{2:x}", can_interface_, motor_id_, (uint32_t)error_id_);
        }
    }
    motor_pos_ =
        range_map(pos_int, uint16_t(0), bitmax<uint16_t>(16), -limit_param_.PosMax, limit_param_.PosMax);
    motor_spd_ =
        range_map(spd_int, uint16_t(0), bitmax<uint16_t>(12), -limit_param_.SpdMax, limit_param_.SpdMax);
    motor_current_ =
        range_map(t_int, uint16_t(0), bitmax<uint16_t>(12), -limit_param_.TauMax, limit_param_.TauMax);
    mos_temperature_ = rx_frame.data[6];
    motor_temperature_ = rx_frame.data[7];
}

void DmMotorDriver::get_motor_param(uint8_t param_cmd) {
    can_frame tx_frame;
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = motor_id_ & 0xFF;
    tx_frame.data[1] = motor_id_ >> 8;
    tx_frame.data[2] = 0x33;
    tx_frame.data[3] = param_cmd;

    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    tx_frame.data[7] = 0xFF;
    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

void DmMotorDriver::motor_pos_cmd(float pos, float spd, bool ignore_limit) {
    if (motor_control_mode_ != POS) {
        set_motor_control_mode(POS);
        return;
    }
    can_frame tx_frame;
    tx_frame.can_id = 0x100 + motor_id_;
    tx_frame.can_dlc = 0x08;
    uint8_t *pbuf, *vbuf;

    spd = limit(spd, -limit_param_.SpdMax, limit_param_.SpdMax);
    pos = limit(pos, -limit_param_.PosMax, limit_param_.PosMax);

    pbuf = (uint8_t*)&pos;
    vbuf = (uint8_t*)&spd;

    tx_frame.data[0] = *pbuf;
    tx_frame.data[1] = *(pbuf + 1);
    tx_frame.data[2] = *(pbuf + 2);
    tx_frame.data[3] = *(pbuf + 3);
    tx_frame.data[4] = *vbuf;
    tx_frame.data[5] = *(vbuf + 1);
    tx_frame.data[6] = *(vbuf + 2);
    tx_frame.data[7] = *(vbuf + 3);

    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

void DmMotorDriver::motor_spd_cmd(float spd) {
    if (motor_control_mode_ != SPD) {
        set_motor_control_mode(SPD);
        return;
    }
    can_frame tx_frame;
    tx_frame.can_id = 0x200 + motor_id_;
    tx_frame.can_dlc = 0x04;

    spd = limit(spd, -limit_param_.SpdMax, limit_param_.SpdMax);
    union32_t rv_type_convert;
    rv_type_convert.f = spd;
    tx_frame.data[0] = rv_type_convert.buf[0];
    tx_frame.data[1] = rv_type_convert.buf[1];
    tx_frame.data[2] = rv_type_convert.buf[2];
    tx_frame.data[3] = rv_type_convert.buf[3];

    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

// Transmit MIT-mDme control(hybrid) package. Called in canTask.
void DmMotorDriver::motor_mit_cmd(float f_p, float f_v, float f_kp, float f_kd, float f_t) {
    if (motor_control_mode_ != MIT) {
        set_motor_control_mode(MIT);
        return;
    }
    uint16_t p, v, kp, kd, t;
    can_frame tx_frame;

    f_p = limit(f_p, -limit_param_.PosMax, limit_param_.PosMax);
    f_v = limit(f_v, -limit_param_.SpdMax, limit_param_.SpdMax);
    f_kp = limit(f_kp, 0.0f, limit_param_.OKpMax);
    f_kd = limit(f_kd, 0.0f, limit_param_.OKdMax);
    f_t = limit(f_t, -limit_param_.TauMax, limit_param_.TauMax);

    p = range_map(f_p, -limit_param_.PosMax, limit_param_.PosMax, uint16_t(0), bitmax<uint16_t>(16));
    v = range_map(f_v, -limit_param_.SpdMax, limit_param_.SpdMax, uint16_t(0), bitmax<uint16_t>(12));
    kp = range_map(f_kp, 0.0f, limit_param_.OKpMax, uint16_t(0), bitmax<uint16_t>(12));
    kd = range_map(f_kd, 0.0f, limit_param_.OKdMax, uint16_t(0), bitmax<uint16_t>(12));
    t = range_map(f_t, -limit_param_.TauMax, limit_param_.TauMax, uint16_t(0), bitmax<uint16_t>(12));

    tx_frame.can_id = motor_id_;
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = p >> 8;
    tx_frame.data[1] = p & 0xFF;
    tx_frame.data[2] = v >> 4;
    tx_frame.data[3] = (v & 0x0F) << 4 | kp >> 8;
    tx_frame.data[4] = kp & 0xFF;
    tx_frame.data[5] = kd >> 4;
    tx_frame.data[6] = (kd & 0x0F) << 4 | t >> 8;
    tx_frame.data[7] = t & 0xFF;

    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

void DmMotorDriver::set_motor_control_mode(uint8_t motor_control_mode) {
    write_register_dm(10, motor_control_mode);
    motor_control_mode_ = motor_control_mode;
}

void DmMotorDriver::set_motor_zero_dm() {
    can_frame tx_frame;
    tx_frame.can_id = motor_id_;  // change according to the mode
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = 0xFF;
    tx_frame.data[1] = 0xFF;
    tx_frame.data[2] = 0xFF;
    tx_frame.data[3] = 0xFF;
    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    tx_frame.data[7] = 0xFE;
    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

void DmMotorDriver::clear_motor_error_dm() {
    can_frame tx_frame;
    tx_frame.can_id = motor_id_;  // change according to the mode
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = 0xFF;
    tx_frame.data[1] = 0xFF;
    tx_frame.data[2] = 0xFF;
    tx_frame.data[3] = 0xFF;
    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    tx_frame.data[7] = 0xFB;
    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

void DmMotorDriver::write_register_dm(uint8_t rid, float value) {
    param_cmd_flag_[rid] = false;
    can_frame tx_frame;
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x08;

    uint8_t* vbuf;
    vbuf = (uint8_t*)&value;

    tx_frame.data[0] = motor_id_ & 0xFF;
    tx_frame.data[1] = motor_id_ >> 8;
    tx_frame.data[2] = 0x55;
    tx_frame.data[3] = rid;

    tx_frame.data[4] = *vbuf;
    tx_frame.data[5] = *(vbuf + 1);
    tx_frame.data[6] = *(vbuf + 2);
    tx_frame.data[7] = *(vbuf + 3);
    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

void DmMotorDriver::write_register_dm(uint8_t rid, int32_t value) {
    param_cmd_flag_[rid] = false;
    can_frame tx_frame;
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x08;

    uint8_t* vbuf;
    vbuf = (uint8_t*)&value;

    tx_frame.data[0] = motor_id_ & 0xFF;
    tx_frame.data[1] = motor_id_ >> 8;
    tx_frame.data[2] = 0x55;
    tx_frame.data[3] = rid;

    tx_frame.data[4] = *vbuf;
    tx_frame.data[5] = *(vbuf + 1);
    tx_frame.data[6] = *(vbuf + 2);
    tx_frame.data[7] = *(vbuf + 3);
    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

void DmMotorDriver::save_register_dm(uint8_t rid) {
    can_frame tx_frame;
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = motor_id_ & 0xFF;
    tx_frame.data[1] = motor_id_ >> 8;
    tx_frame.data[2] = 0xAA;
    tx_frame.data[3] = rid;

    tx_frame.data[4] = 0xFF;
    tx_frame.data[5] = 0xFF;
    tx_frame.data[6] = 0xFF;
    tx_frame.data[7] = 0xFF;
    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

void DmMotorDriver::refresh_motor_status() {
    can_frame tx_frame;
    tx_frame.can_id = 0x7FF;
    tx_frame.can_dlc = 0x08;

    tx_frame.data[0] = motor_id_ & 0xFF;
    tx_frame.data[1] = motor_id_ >> 8;
    tx_frame.data[2] = 0xCC;
    tx_frame.data[3] = 0x00;

    tx_frame.data[4] = 0x00;
    tx_frame.data[5] = 0x00;
    tx_frame.data[6] = 0x00;
    tx_frame.data[7] = 0x00;
    can_->transmit(tx_frame);
    {
        response_count_++;
    }
}

void DmMotorDriver::clear_motor_error() {
    clear_motor_error_dm();
}