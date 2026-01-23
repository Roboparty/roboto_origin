#include "canopen_parser.h"
#include <stdint.h>

#define TPDO1_BASE  0x180
#define TPDO2_BASE  0x280
#define TPDO3_BASE  0x380
#define TPDO4_BASE  0x480
#define TPDO6_BASE  0x680
#define TPDO7_BASE  0x780

static int parse_canopen_accel(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int16_t *raw = (int16_t*)frame->data;
    data->acc_x = raw[0] * 0.001f * 9.81f;
    data->acc_y = raw[1] * 0.001f * 9.81f;
    data->acc_z = raw[2] * 0.001f * 9.81f;
    return CAN_MSG_ACCEL;
}

static int parse_canopen_gyro(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int16_t *raw = (int16_t*)frame->data;
    data->gyr_x = raw[0] * 0.1f * 0.017453f;
    data->gyr_y = raw[1] * 0.1f * 0.017453f;
    data->gyr_z = raw[2] * 0.1f * 0.017453f;
    return CAN_MSG_GYRO;
}

static int parse_canopen_euler(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int16_t *raw = (int16_t*)frame->data;
    data->roll = raw[0] * 0.01f * 0.017453f;
    data->pitch = raw[1] * 0.01f * 0.017453f;
    data->imu_yaw = raw[2] * 0.01f * 0.017453f;
    return CAN_MSG_EULER;
}

static int parse_canopen_quat(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int16_t *raw = (int16_t*)frame->data;
    data->quat_w = raw[0] * 0.0001f;
    data->quat_x = raw[1] * 0.0001f;
    data->quat_y = raw[2] * 0.0001f;
    data->quat_z = raw[3] * 0.0001f;
    return CAN_MSG_QUAT;
}

static int parse_canopen_pressure(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int32_t *raw = (int32_t*)frame->data;
    data->pressure = *raw;
    return CAN_MSG_PRESSURE;
}

static int parse_canopen_inclination(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    int32_t *raw = (int32_t*)frame->data;
    data->incli_x = raw[0] * 0.01f;
    data->incli_y = raw[1] * 0.01f;
    return CAN_MSG_INCLI;
}

int canopen_parse_frame(const hipnuc_can_frame_t *frame, can_sensor_data_t *data)
{
    if (!frame || !data) return CAN_MSG_ERROR;

    uint32_t id = frame->can_id & HIPNUC_CAN_SFF_MASK;
    uint32_t base_id = id & 0x780;

    switch (base_id) {
        case TPDO1_BASE: return parse_canopen_accel(frame, data);
        case TPDO2_BASE: return parse_canopen_gyro(frame, data);
        case TPDO3_BASE: return parse_canopen_euler(frame, data);
        case TPDO4_BASE: return parse_canopen_quat(frame, data);
        case TPDO6_BASE: return parse_canopen_pressure(frame, data);
        case TPDO7_BASE: return parse_canopen_inclination(frame, data);
        default:         return CAN_MSG_UNKNOWN;
    }
}
