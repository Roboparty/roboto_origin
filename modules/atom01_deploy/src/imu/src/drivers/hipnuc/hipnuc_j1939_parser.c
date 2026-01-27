#include "hipnuc_j1939_parser.h"

#define J1939_PGN_LONLAT     0xFF10
#define J1939_PGN_ALT        0xFF14
#define J1939_PGN_STATUS     0xFF18
#define J1939_PGN_VEL        0xFF26
#define J1939_PGN_TIME       0xFF2F
#define J1939_PGN_ACCEL      0xFF34
#define J1939_PGN_GYRO       0xFF37
#define J1939_PGN_MAG        0xFF3A
#define J1939_PGN_PITCH_ROLL 0xFF3D
#define J1939_PGN_YAW        0xFF41
#define J1939_PGN_ENV        0xFF43
#define J1939_PGN_QUAT       0xFF46
#define J1939_PGN_INCLINE    0xFF4A

static int parse_j1939_time(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    const uint8_t *raw = frame->data;
    data->utc_year = raw[0];
    data->utc_month = raw[1];
    data->utc_day = raw[2];
    data->hours = raw[3];
    data->minutes = raw[4];
    data->seconds = raw[5];
    data->milliseconds = raw[6] | (raw[7] << 8);
    data->timestamp_ms = (uint32_t)data->hours * 3600000UL + (uint32_t)data->minutes * 60000UL + (uint32_t)data->seconds * 1000UL + data->milliseconds;
    return CAN_MSG_TIME;
}

static int parse_j1939_accel(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int16_t *raw = (int16_t*)frame->data;
    data->acc_x = raw[0] * 0.00048828f;
    data->acc_y = raw[1] * 0.00048828f;
    data->acc_z = raw[2] * 0.00048828f;
    return CAN_MSG_ACCEL;
}

static int parse_j1939_gyro(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int16_t *raw = (int16_t*)frame->data;
    data->gyr_x = raw[0] * 0.061035f;
    data->gyr_y = raw[1] * 0.061035f;
    data->gyr_z = raw[2] * 0.061035f;
    return CAN_MSG_GYRO;
}

static int parse_j1939_mag(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int16_t *raw = (int16_t*)frame->data;
    data->mag_x = raw[0] * 0.030517f;
    data->mag_y = raw[1] * 0.030517f;
    data->mag_z = raw[2] * 0.030517f;
    return CAN_MSG_MAG;
}

static int parse_j1939_pitch_roll(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int32_t *raw = (int32_t*)frame->data;
    data->roll = raw[0] * 0.001f;
    data->pitch = raw[1] * 0.001f;
    return CAN_MSG_PITCH_ROLL;
}

static int parse_j1939_yaw(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int32_t *raw = (int32_t*)frame->data;
    data->imu_yaw = raw[0] * 0.001f;
    return CAN_MSG_YAW;
}

static int parse_j1939_quat(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int16_t *raw = (int16_t*)frame->data;
    data->quat_w = raw[0] * 0.0001f;
    data->quat_x = raw[1] * 0.0001f;
    data->quat_y = raw[2] * 0.0001f;
    data->quat_z = raw[3] * 0.0001f;
    return CAN_MSG_QUAT;
}

static int parse_j1939_inclination(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int32_t *raw = (int32_t*)frame->data;
    data->incli_x = raw[0] * 0.001f;
    data->incli_y = raw[1] * 0.001f;
    return CAN_MSG_INCLI;
}

static int parse_j1939_env(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int16_t *raw16 = (int16_t*)frame->data;
    int32_t *raw32 = (int32_t*)(frame->data + 4);
    data->temperature = raw16[0] * 0.01f;
    data->pressure = raw32[0] * 0.01f;
    return CAN_MSG_PRESSURE;
}

static int parse_j1939_lonlat(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int32_t *raw = (int32_t*)frame->data;
    data->ins_lat = raw[0] * 1e-7;
    data->ins_lon = raw[1] * 1e-7;
    return CAN_MSG_GNSS_POS;
}

static int parse_j1939_alt(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int32_t *raw32 = (int32_t*)frame->data;
    int16_t *raw16 = (int16_t*)frame->data;
    data->ins_msl = raw32[0] * 0.01;
    data->undulation = raw16[2] * 0.01f;
    data->diff_age_s = raw16[3] * 0.01f;
    return CAN_MSG_GNSS_POS;
}

static int parse_j1939_status(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    const uint8_t *raw = frame->data;
    data->solq_pos = raw[0];
    data->solq_heading = raw[1];
    data->nv_pos = raw[2];
    data->nv_heading = raw[3];
    data->ins_status = raw[4];
    return CAN_MSG_GNSS_STATUS;
}

static int parse_j1939_vel(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int16_t *raw = (int16_t*)frame->data;
    data->ins_vel_e = raw[0] * 0.01f;
    data->ins_vel_n = raw[1] * 0.01f;
    data->ins_vel_u = raw[2] * 0.01f;
    data->ins_speed = raw[3] * 0.01f;
    return CAN_MSG_GNSS_VEL;
}

int hipnuc_j1939_parse_frame(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    if (!frame || !data) return CAN_MSG_ERROR;
    if (!(frame->can_id & HIPNUC_CAN_EFF_FLAG)) return CAN_MSG_UNKNOWN;
    uint32_t id = frame->can_id & HIPNUC_CAN_EFF_MASK;
    uint32_t pgn = (id >> 8) & 0xFFFF;
    switch (pgn) {
        case J1939_PGN_LONLAT:     return parse_j1939_lonlat(frame, data);
        case J1939_PGN_ALT:        return parse_j1939_alt(frame, data);
        case J1939_PGN_STATUS:     return parse_j1939_status(frame, data);
        case J1939_PGN_VEL:        return parse_j1939_vel(frame, data);
        case J1939_PGN_TIME:       return parse_j1939_time(frame, data);
        case J1939_PGN_ACCEL:      return parse_j1939_accel(frame, data);
        case J1939_PGN_GYRO:       return parse_j1939_gyro(frame, data);
        case J1939_PGN_MAG:        return parse_j1939_mag(frame, data);
        case J1939_PGN_PITCH_ROLL: return parse_j1939_pitch_roll(frame, data);
        case J1939_PGN_YAW:        return parse_j1939_yaw(frame, data);
        case J1939_PGN_ENV:        return parse_j1939_env(frame, data);
        case J1939_PGN_QUAT:       return parse_j1939_quat(frame, data);
        case J1939_PGN_INCLINE:    return parse_j1939_inclination(frame, data);
        default:                   return CAN_MSG_UNKNOWN;
    }
}
