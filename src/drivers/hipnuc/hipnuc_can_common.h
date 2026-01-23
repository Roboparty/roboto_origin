#ifndef D515A6BC_1F44_4C6C_A130_CB9A79773892
#define D515A6BC_1F44_4C6C_A130_CB9A79773892
#ifndef HIPNUC_CAN_COMMON_H
#define HIPNUC_CAN_COMMON_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint32_t can_id;
    uint8_t can_dlc;
    uint8_t data[8];
    uint64_t hw_ts_us;
} hipnuc_can_frame_t;

#define HIPNUC_CAN_EFF_FLAG    0x80000000U
#define HIPNUC_CAN_SFF_MASK    0x000007FFU
#define HIPNUC_CAN_EFF_MASK    0x1FFFFFFFU

#define CAN_MSG_ERROR       0
#define CAN_MSG_ACCEL       1
#define CAN_MSG_GYRO        2
#define CAN_MSG_MAG         3
#define CAN_MSG_TEMP        4
#define CAN_MSG_QUAT        5
#define CAN_MSG_EULER       6
#define CAN_MSG_PRESSURE    7
#define CAN_MSG_GNSS_POS    8
#define CAN_MSG_GNSS_VEL    9
#define CAN_MSG_INCLI       10
#define CAN_MSG_TIME        11
#define CAN_MSG_PITCH_ROLL  12
#define CAN_MSG_YAW         13
#define CAN_MSG_GNSS_STATUS 14
#define CAN_MSG_UNKNOWN     99

typedef struct {
    uint8_t node_id;
    uint64_t hw_ts_us;
    float acc_x;
    float acc_y;
    float acc_z;
    float gyr_x;
    float gyr_y;
    float gyr_z;
    float mag_x;
    float mag_y;
    float mag_z;
    float quat_w;
    float quat_x;
    float quat_y;
    float quat_z;
    float roll;
    float pitch;
    float imu_yaw;
    float incli_x;
    float incli_y;
    float temperature;
    float pressure;
    uint8_t utc_year;
    uint8_t utc_month;
    uint8_t utc_day;
    uint8_t hours;
    uint8_t minutes;
    uint8_t seconds;
    uint16_t milliseconds;
    uint32_t timestamp_ms;
    double ins_lat;
    double ins_lon;
    double ins_msl;
    float undulation;
    float diff_age_s;
    float ins_vel_e;
    float ins_vel_n;
    float ins_vel_u;
    float ins_speed;
    uint8_t solq_pos;
    uint8_t solq_heading;
    uint8_t nv_pos;
    uint8_t nv_heading;
    uint8_t ins_status;
} can_sensor_data_t;

typedef struct {
    char buffer[512];
    size_t length;
} can_json_output_t;

int hipnuc_can_to_json(const can_sensor_data_t *data, int msg_type, can_json_output_t *output);
uint8_t hipnuc_can_extract_node_id(uint32_t can_id);

typedef enum {
    HIPNUC_J1939_CMD_READ = 0x03,
    HIPNUC_J1939_CMD_WRITE = 0x06
} hipnuc_j1939_cmd_t;

uint32_t hipnuc_j1939_make_cfg_id(uint8_t da, uint8_t sa);
void hipnuc_j1939_build_cfg_write(uint8_t da, uint8_t sa, uint16_t addr, uint32_t val, hipnuc_can_frame_t *out);
void hipnuc_j1939_build_cfg_read(uint8_t da, uint8_t sa, uint16_t addr, uint32_t len_regs, hipnuc_can_frame_t *out);
int hipnuc_j1939_parse_cfg(const hipnuc_can_frame_t *frame, uint16_t *addr, hipnuc_j1939_cmd_t *cmd, uint8_t *status, uint32_t *val);


#endif


#endif /* D515A6BC_1F44_4C6C_A130_CB9A79773892 */
