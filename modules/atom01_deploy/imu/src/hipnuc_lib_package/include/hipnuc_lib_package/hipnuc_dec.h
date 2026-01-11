/*
 * Copyright (c) 2006-2024, HiPNUC
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * HiPNUC Decoder Library
 * 
 * This file contains the declarations for the HiPNUC decoder functions and data structures.
 * It provides an interface for decoding HiPNUC protocol messages, including IMU and INS data.
 */

 #ifndef __HIPNUC_DEC_H__
 #define __HIPNUC_DEC_H__
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 #include <stdint.h>
 #include <stdio.h>
 #include <string.h>
 
 #ifdef QT_CORE_LIB
 #pragma pack(push)
 #pragma pack(1)
 #endif
 
 /* HiPNUC protocol constants */
 #define HIPNUC_MAX_RAW_SIZE     (256)   /* Maximum size of raw message buffer */
 
 /**
  * Packet 0x91: IMU data (floating point)
  */
 typedef struct __attribute__((__packed__))
 {
     uint8_t         tag;            /* Data packet tag, if tag = 0x00, means that this packet is null */
     uint16_t        sttaus;         /* reserved */
     int8_t          temp;           /* Temperature */
     float           air_pressure;   /* Pressure */
     uint32_t        system_time;    /* Timestamp */
     float           acc[3];         /* Accelerometer data (x, y, z) */
     float           gyr[3];         /* Gyroscope data (x, y, z) */
     float           mag[3];         /* Magnetometer data (x, y, z) */
     float           roll;           /* Roll angle */
     float           pitch;          /* Pitch angle */
     float           yaw;            /* Yaw angle */
     float           quat[4];        /* Quaternion (w, x, y, z) */
 } hi91_t;
 
 /**
  * Packet 0x92: IMU data (integer type)
  */
 typedef struct __attribute__((__packed__))
 {
     uint8_t         tag;            /* Data packet tag */
     uint16_t        status;         /* Status information */
     int8_t          temperature;    /* Temperature */
     uint16_t        rev;            /* reserved */
     int16_t         air_pressure;   /* Air pressure */
     int16_t         reserved;       /* Reserved field */
     int16_t         gyr_b[3];       /* Gyroscope data (raw) */
     int16_t         acc_b[3];       /* Accelerometer data (raw) */
     int16_t         mag_b[3];       /* Magnetometer data (raw) */
     int32_t         roll;           /* Roll angle (raw) */
     int32_t         pitch;          /* Pitch angle (raw) */
     int32_t         yaw;            /* Yaw angle (raw) */
     int16_t         quat[4];        /* Quaternion (raw) */
 } hi92_t;
 
 /**
  * Packet 0x81: INS data, including lat, lon, eul, quat, raw IMU data 
  */
 typedef struct __attribute__((__packed__))
 {
     uint8_t         tag;            /* Data packet tag */
     uint16_t        status;         /* Status information */
     uint8_t         ins_status;     /* INS status */
     uint16_t        gpst_wn;        /* GPS time: week number */
     uint32_t        gpst_tow;       /* GPS time: time of week */
     uint16_t        reserved;       /* reserved */
     int16_t         gyr_b[3];       /* Gyroscope data (raw) */
     int16_t         acc_b[3];       /* Accelerometer data (raw) */
     int16_t         mag_b[3];       /* Magnetometer data (raw) */
     int16_t         air_pressure;   /* Air pressure */
     int16_t         reserved1;      /* Reserved field */
     int8_t          temperature;    /* Temperature */
     uint8_t         utc_year;       /* UTC year */
     uint8_t         utc_month;      /* UTC month */
     uint8_t         utc_day;        /* UTC day */
     uint8_t         utc_hour;       /* UTC hour */
     uint8_t         utc_min;        /* UTC minute */
     uint16_t        utc_msec;       /* UTC milliseconds */
     int16_t         roll;           /* Roll angle */
     int16_t         pitch;          /* Pitch angle */
     uint16_t        yaw;            /* Yaw angle */
     int16_t         quat[4];        /* Quaternion */
     int32_t         ins_lon;        /* INS longitude */
     int32_t         ins_lat;        /* INS latitude */
     int32_t         ins_msl;        /* INS mean sea level altitude */
     uint8_t         pdop;           /* Position dilution of precision */
     uint8_t         hdop;           /* Horizontal dilution of precision */
     uint8_t         solq_pos;       /* Solution quality for position */
     uint8_t         nv_pos;         /* Number of satellites used for position */
     uint8_t         solq_heading;   /* Solution quality for heading */
     uint8_t         nv_heading;     /* Number of satellites used for heading */
     uint8_t         diff_age;       /* Differential age */
     int16_t         undulation;     /* Undulation */
     uint8_t         ant_status;     /* Reserved field */
     int16_t         vel_enu[3];     /* Velocity in ENU frame */
     int16_t         acc_enu[3];     /* Acceleration in ENU frame */
     int32_t         gnss_lon;       /* GNSS longitude */
     int32_t         gnss_lat;       /* GNSS latitude */
     int32_t         gnss_msl;       /* GNSS mean sea level altitude */
     uint8_t         reserved2[2];   /* Reserved field */
 } hi81_t;
 
 /**
  * HiPNUC raw data structure
  */
 typedef struct
 {
     int nbyte;                          /* Number of bytes in message buffer */ 
     int len;                            /* Message length (bytes) */
     uint8_t buf[HIPNUC_MAX_RAW_SIZE];   /* Message raw buffer */
     hi91_t hi91;                        /* Decoded 0x91 packet data */
     hi92_t hi92;                        /* Decoded 0x92 packet data */
     hi81_t hi81;                        /* Decoded 0x81 packet data */
 } hipnuc_raw_t;
 
 #ifdef QT_CORE_LIB
 #pragma pack(pop)
 #endif
 
 /**
  * @brief Process one byte of input data for HiPNUC decoder
  *
  * @param raw Pointer to hipnuc_raw_t structure
  * @param data Input byte to process
  * @return int 1 if a complete packet was successfully decoded, 0 if more data is needed, -1 on error
  */
 int hipnuc_input(hipnuc_raw_t *raw, uint8_t data);
 
 /**
  * @brief Dump decoded HiPNUC packet data to a string buffer
  *
  * @param raw Pointer to hipnuc_raw_t structure containing decoded data
  * @param buf Output buffer to store the formatted string
  * @param buf_size Size of the output buffer
  * @return int Number of characters written to the buffer
  */
 int hipnuc_dump_packet(hipnuc_raw_t *raw, char *buf, size_t buf_size);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif /* __HIPNUC_DEC_H__ */
 