#include "hipnuc_can_common.h"
#include <stdarg.h>
#include <stdio.h>

static void json_append(can_json_output_t *out, const char *fmt, ...)
{
    if (out->length >= sizeof(out->buffer) - 1) return;
    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(out->buffer + out->length, sizeof(out->buffer) - out->length, fmt, args);
    va_end(args);
    if (written > 0) out->length += written;
}

int hipnuc_can_to_json(const can_sensor_data_t *data, int msg_type, can_json_output_t *output)
{
    if (!data || !output) return -1;
    output->length = 0;
    output->buffer[0] = '\0';
    json_append(output, "{\"node_id\":%d,\"hw_ts_us\":%llu,\"data\":{", data->node_id, (unsigned long long)data->hw_ts_us);
    switch (msg_type) {
        case CAN_MSG_ACCEL:
            json_append(output, "\"acc_x\":%.6f,\"acc_y\":%.6f,\"acc_z\":%.6f", data->acc_x, data->acc_y, data->acc_z);
            break;
        case CAN_MSG_GYRO:
            json_append(output, "\"gyr_x\":%.6f,\"gyr_y\":%.6f,\"gyr_z\":%.6f", data->gyr_x, data->gyr_y, data->gyr_z);
            break;
        case CAN_MSG_MAG:
            json_append(output, "\"mag_x\":%.3f,\"mag_y\":%.3f,\"mag_z\":%.3f", data->mag_x, data->mag_y, data->mag_z);
            break;
        case CAN_MSG_QUAT:
            json_append(output, "\"quat_w\":%.4f,\"quat_x\":%.4f,\"quat_y\":%.4f,\"quat_z\":%.4f", data->quat_w, data->quat_x, data->quat_y, data->quat_z);
            break;
        case CAN_MSG_EULER:
        case CAN_MSG_PITCH_ROLL:
            json_append(output, "\"roll\":%.6f,\"pitch\":%.6f", data->roll, data->pitch);
            if (msg_type == CAN_MSG_EULER) json_append(output, ",\"imu_yaw\":%.6f", data->imu_yaw);
            break;
        case CAN_MSG_YAW:
            json_append(output, "\"imu_yaw\":%.6f", data->imu_yaw);
            break;
        case CAN_MSG_PRESSURE:
            json_append(output, "\"temperature\":%.2f,\"pressure\":%.2f", data->temperature, data->pressure);
            break;
        case CAN_MSG_INCLI:
            json_append(output, "\"incli_x\":%.2f,\"incli_y\":%.2f", data->incli_x, data->incli_y);
            break;
        case CAN_MSG_TIME:
            json_append(output, "\"utc_year\":%d,\"utc_month\":%d,\"utc_day\":%d,\"hours\":%d,\"minutes\":%d,\"seconds\":%d,\"milliseconds\":%d,\"timestamp_ms\":%u", data->utc_year, data->utc_month, data->utc_day, data->hours, data->minutes, data->seconds, data->milliseconds, data->timestamp_ms);
            break;
        case CAN_MSG_GNSS_POS:
            json_append(output, "\"ins_lon\":%.8f,\"ins_lat\":%.8f,\"ins_msl\":%.3f,\"undulation\":%.2f,\"diff_age_s\":%.2f", data->ins_lon, data->ins_lat, data->ins_msl, data->undulation, data->diff_age_s);
            break;
        case CAN_MSG_GNSS_VEL:
            json_append(output, "\"ins_vel_e\":%.3f,\"ins_vel_n\":%.3f,\"ins_vel_u\":%.3f,\"ins_speed\":%.3f", data->ins_vel_e, data->ins_vel_n, data->ins_vel_u, data->ins_speed);
            break;
        case CAN_MSG_GNSS_STATUS:
            json_append(output, "\"solq_pos\":%d,\"solq_heading\":%d,\"nv_pos\":%d,\"nv_heading\":%d,\"ins_status\":%d", data->solq_pos, data->solq_heading, data->nv_pos, data->nv_heading, data->ins_status);
            break;
        default:
            json_append(output, "\"error\":\"unknown message type\"");
            break;
    }
    json_append(output, "}}\n");
    return output->length;
}

uint8_t hipnuc_can_extract_node_id(uint32_t can_id)
{
    if (can_id & HIPNUC_CAN_EFF_FLAG) return (uint8_t)(can_id & 0xFF);
    return (uint8_t)(can_id & 0x7F);
}

static void le16_encode(uint16_t v, uint8_t *o)
{
    o[0] = (uint8_t)(v & 0xFF);
    o[1] = (uint8_t)((v >> 8) & 0xFF);
}

static void le32_encode(uint32_t v, uint8_t *o)
{
    o[0] = (uint8_t)(v & 0xFF);
    o[1] = (uint8_t)((v >> 8) & 0xFF);
    o[2] = (uint8_t)((v >> 16) & 0xFF);
    o[3] = (uint8_t)((v >> 24) & 0xFF);
}

uint32_t hipnuc_j1939_make_cfg_id(uint8_t da, uint8_t sa)
{
    return 0x0CEF0000U | ((uint32_t)da << 8) | (uint32_t)sa;
}

void hipnuc_j1939_build_cfg_write(uint8_t da, uint8_t sa, uint16_t addr, uint32_t val, hipnuc_can_frame_t *out)
{
    out->can_id = HIPNUC_CAN_EFF_FLAG | hipnuc_j1939_make_cfg_id(da, sa);
    out->can_dlc = 8;
    le16_encode(addr, &out->data[0]);
    out->data[2] = (uint8_t)HIPNUC_J1939_CMD_WRITE;
    out->data[3] = 0;
    le32_encode(val, &out->data[4]);
    out->hw_ts_us = 0;
}

void hipnuc_j1939_build_cfg_read(uint8_t da, uint8_t sa, uint16_t addr, uint32_t len_regs, hipnuc_can_frame_t *out)
{
    out->can_id = HIPNUC_CAN_EFF_FLAG | hipnuc_j1939_make_cfg_id(da, sa);
    out->can_dlc = 8;
    le16_encode(addr, &out->data[0]);
    out->data[2] = (uint8_t)HIPNUC_J1939_CMD_READ;
    out->data[3] = 0;
    le32_encode(len_regs, &out->data[4]);
    out->hw_ts_us = 0;
}

int hipnuc_j1939_parse_cfg(const hipnuc_can_frame_t *frame, uint16_t *addr, hipnuc_j1939_cmd_t *cmd, uint8_t *status, uint32_t *val)
{
    if (!frame || frame->can_dlc != 8) return -1;
    if (!(frame->can_id & HIPNUC_CAN_EFF_FLAG)) return -1;
    if (((frame->can_id & HIPNUC_CAN_EFF_MASK) >> 16) != 0x0CEF) return -1;
    if (addr) *addr = (uint16_t)(frame->data[0] | ((uint16_t)frame->data[1] << 8));
    if (cmd) *cmd = (hipnuc_j1939_cmd_t)frame->data[2];
    if (status) *status = frame->data[3];
    if (val) {
        *val = (uint32_t)frame->data[4]
             | ((uint32_t)frame->data[5] << 8)
             | ((uint32_t)frame->data[6] << 16)
             | ((uint32_t)frame->data[7] << 24);
    }
    return 0;
}
