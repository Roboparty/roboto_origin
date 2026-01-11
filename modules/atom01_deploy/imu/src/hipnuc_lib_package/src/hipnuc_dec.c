
/*
 * Copyright (c) 2006-2024, HiPNUC
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

 #include "hipnuc_lib_package/hipnuc_dec.h"

 /* The driver file for decoding HiPNUC protocol, DO NOT MODIFTY*/
 
 /* HiPNUC protocol constants */
 #define CHSYNC1                 (0x5A)              /* CHAOHE message sync code 1 */
 #define CHSYNC2                 (0xA5)              /* CHAOHE message sync code 2 */
 #define CH_HDR_SIZE             (0x06)              /* CHAOHE protocol header size */
 
 /* legcy support of HI226/HI229 */
 #define HIPNUC_ID_USRID         (0x90)
 #define HIPNUC_ID_ACC_RAW       (0xA0)
 #define HIPNUC_ID_ACC_CAL       (0xA1)
 #define HIPNUC_ID_GYR_RAW       (0xB0)
 #define HIPNUC_ID_GYR_CAL       (0xB1)
 #define HIPNUC_ID_MAG_RAW       (0xC0)
 #define HIPNUC_ID_EUL           (0xD0)
 #define HIPNUC_ID_QUAT          (0xD1)
 #define HIPNUC_ID_PRS           (0xF0)
 
 /* new HiPNUC standard packet */
 #define HIPNUC_ID_HI91        (0x91)
 #define HIPNUC_ID_HI92        (0x92)
 #define HIPNUC_ID_HI81        (0x81)
 
 #ifndef D2R
 #define D2R (0.0174532925199433F)
 #endif
 
 #ifndef R2D
 #define R2D (57.2957795130823F)
 #endif
 
 #ifndef GRAVITY
 #define GRAVITY (9.8F)
 #endif
 
 
 static void hipnuc_crc16(uint16_t *inital, const uint8_t *buf, uint32_t len);
 
 /* common type conversion */
 #define I2(p) (*((int16_t *)(p)))
 static uint16_t U2(uint8_t *p)
 {
     uint16_t u;
     memcpy(&u, p, 2);
     return u;
 }
 
 static float R4(uint8_t *p)
 {
     float r;
     memcpy(&r, p, 4);
     return r;
 }
 
 /* parse the payload of a frame and feed into data section */
 static int parse_data(hipnuc_raw_t *raw)
 {
     int ofs = 0;
     uint8_t *p = &raw->buf[CH_HDR_SIZE];
     
     /* ignore all previous data */
     raw->hi91.tag = 0;
     raw->hi81.tag = 0;
     raw->hi92.tag = 0;
 
     while (ofs < raw->len)
     {
         switch (p[ofs])
         {
         case HIPNUC_ID_USRID:
             ofs += 2;
             break;
         case HIPNUC_ID_ACC_RAW:
         case HIPNUC_ID_ACC_CAL:
              raw->hi91.tag = HIPNUC_ID_HI91;
              raw->hi91.acc[0] = (float)I2(p + ofs + 1) / 1000;
              raw->hi91.acc[1] = (float)I2(p + ofs + 3) / 1000;
              raw->hi91.acc[2] = (float)I2(p + ofs + 5) / 1000;
             ofs += 7;
             break;
         case HIPNUC_ID_GYR_RAW:
         case HIPNUC_ID_GYR_CAL:
             raw->hi91.tag = HIPNUC_ID_HI91;
             raw->hi91.gyr[0] = (float)I2(p + ofs + 1) / 10;
             raw->hi91.gyr[1] = (float)I2(p + ofs + 3) / 10;
             raw->hi91.gyr[2] = (float)I2(p + ofs + 5) / 10;
             ofs += 7;
             break;
         case HIPNUC_ID_MAG_RAW:
             raw->hi91.tag = HIPNUC_ID_HI91;
             raw->hi91.mag[0] = (float)I2(p + ofs + 1) / 10;
             raw->hi91.mag[1] = (float)I2(p + ofs + 3) / 10;
             raw->hi91.mag[2] = (float)I2(p + ofs + 5) / 10;
             ofs += 7;
             break;
         case HIPNUC_ID_EUL:
             raw->hi91.tag = HIPNUC_ID_HI91;
             raw->hi91.pitch = (float)I2(p + ofs + 1) / 100;
             raw->hi91.roll = (float)I2(p + ofs + 3) / 100;
             raw->hi91.yaw = (float)I2(p + ofs + 5) / 10;
             ofs += 7;
             break;
         case HIPNUC_ID_QUAT:
             raw->hi91.tag = HIPNUC_ID_HI91;
             raw->hi91.quat[0] = R4(p + ofs + 1);
             raw->hi91.quat[1] = R4(p + ofs + 5);
             raw->hi91.quat[2] = R4(p + ofs + 9);
             raw->hi91.quat[3] = R4(p + ofs + 13);
             ofs += 17;
             break;
         case HIPNUC_ID_PRS:
             raw->hi91.tag = HIPNUC_ID_HI91;
             raw->hi91.air_pressure = R4(p + ofs + 1);
             ofs += 5;
             break;
         case HIPNUC_ID_HI91:
             memcpy(&raw->hi91, p + ofs, sizeof(hi91_t));
             ofs += sizeof(hi91_t);
             break;
         case HIPNUC_ID_HI81:
             memcpy(&raw->hi81, p + ofs, sizeof(hi81_t));
             ofs += sizeof(hi81_t);
             break;
         case HIPNUC_ID_HI92:
             memcpy(&raw->hi92, p + ofs, sizeof(hi92_t));
             ofs += sizeof(hi92_t);
             break;
         default:
             ofs++;
             break;
         }
     }
     return 1;
 }
 
 static int decode_hipnuc(hipnuc_raw_t *raw)
 {
     uint16_t crc = 0;
 
     /* checksum */
     hipnuc_crc16(&crc, raw->buf, (CH_HDR_SIZE-2));
     hipnuc_crc16(&crc, raw->buf + CH_HDR_SIZE, raw->len);
     if (crc != U2(raw->buf + (CH_HDR_SIZE-2)))
     {
         // NL_TRACE("ch checksum error: frame:0x%X calcuate:0x%X, len:%d\n", U2(raw->buf + 4), crc, raw->len);
         return -1;
     }
 
     return parse_data(raw);
 }
 
 /* sync code */
 static int sync_hipnuc(uint8_t *buf, uint8_t data)
 {
     buf[0] = buf[1];
     buf[1] = data;
     return buf[0] == CHSYNC1 && buf[1] == CHSYNC2;
 }
 
 /**
  * @brief     HiPNUC decoder input, read one byte at a time.
  *
  * @param    raw is the decoder struct.
  * @param    data is the one byte read from stream.
  * @return   >0: decoder received a frame successfully, else: receiver did not receive a frame successfully.
  */
 int hipnuc_input(hipnuc_raw_t *raw, uint8_t data)
 {
     /* synchronize frame */
     if (raw->nbyte == 0)
     {
         if (!sync_hipnuc(raw->buf, data))
             return 0;
         raw->nbyte = 2;
         return 0;
     }
 
     raw->buf[raw->nbyte++] = data;
 
     if (raw->nbyte == CH_HDR_SIZE)
     {
         if ((raw->len = U2(raw->buf + 2)) > (HIPNUC_MAX_RAW_SIZE - CH_HDR_SIZE))
         {
             // NL_TRACE("ch length error: len=%d\n",raw->len);
             raw->nbyte = 0;
             return -1;
         }
     }
 
     if (raw->nbyte < CH_HDR_SIZE || raw->nbyte < (raw->len + CH_HDR_SIZE))
     {
         return 0;
     }
 
     raw->nbyte = 0;
 
     return decode_hipnuc(raw);
 }
 
 
 /**
  * @brief    Convert packet to string, only dump parts of data
  *
  * @param    raw is struct of decoder
  * @param    buf is the log string buffer, make sure buf is larger than 256
  * @param    buf_size is the size of the log buffer
  * @return   Number of characters written to the buffer
  */
 int hipnuc_dump_packet(hipnuc_raw_t *raw, char *buf, size_t buf_size)
 {
     int written = 0;
     int ret;
 
     /* dump 0x91 packet */
     if(raw->hi91.tag == HIPNUC_ID_HI91)
     {
         /* Format:
          * system_time: ms
          * acc: m/s²
          * gyr: deg/s
          * mag: uT
          * pitch/roll/yaw: deg
          * quat: w,x,y,z
          * air_pressure: Pa
          */
         ret = snprintf(buf + written, buf_size - written,
             "{\n"
             "  \"type\": \"HI91\",\n"
             "  \"system_time\": %d,\n"
             "  \"acc\": [%.3f, %.3f, %.3f],\n"
             "  \"gyr\": [%.3f, %.3f, %.3f],\n"
             "  \"mag\": [%.3f, %.3f, %.3f],\n"
             "  \"pitch\": %.2f,\n"
             "  \"roll\": %.2f,\n"
             "  \"yaw\": %.2f,\n"
             "  \"quat\": [%.3f, %.3f, %.3f, %.3f],\n"
             "  \"air_pressure\": %.1f\n"
             "}\n",
             raw->hi91.system_time,
             raw->hi91.acc[0]*GRAVITY, raw->hi91.acc[1]*GRAVITY, raw->hi91.acc[2]*GRAVITY,
             raw->hi91.gyr[0], raw->hi91.gyr[1], raw->hi91.gyr[2],
             raw->hi91.mag[0], raw->hi91.mag[1], raw->hi91.mag[2],
             raw->hi91.pitch, raw->hi91.roll, raw->hi91.yaw,
             raw->hi91.quat[0], raw->hi91.quat[1], raw->hi91.quat[2], raw->hi91.quat[3],
             raw->hi91.air_pressure);
     }
     
     /* dump 0x92 packet */
     else if(raw->hi92.tag == HIPNUC_ID_HI92)
     {
         /* Format:
          * temperature: °C
          * acc: m/s²
          * gyr: deg/s
          * mag: uT
          * pitch/roll/yaw: deg
          */
         ret = snprintf(buf + written, buf_size - written,
             "{\n"
             "  \"type\": \"HI92\",\n"
             "  \"status\": %d,\n"
             "  \"temperature\": %d,\n"
             "  \"acc\": [%.3f, %.3f, %.3f],\n"
             "  \"gyr\": [%.3f, %.3f, %.3f],\n"
             "  \"mag\": [%.3f, %.3f, %.3f],\n"
             "  \"pitch\": %.2f,\n"
             "  \"roll\": %.2f,\n"
             "  \"yaw\": %.2f\n"
             "  \"quat\": [%.3f, %.3f, %.3f, %.3f],\n"
             "}\n",
             raw->hi92.status,
             raw->hi92.temperature,
             raw->hi92.acc_b[0]*0.0048828, raw->hi92.acc_b[1]*0.0048828, raw->hi92.acc_b[2]*0.0048828,
             raw->hi92.gyr_b[0]*(0.001*R2D), raw->hi92.gyr_b[1]*(0.001*R2D), raw->hi92.gyr_b[2]*(0.001*R2D),
             raw->hi92.mag_b[0]*0.030517, raw->hi92.mag_b[1]*0.030517, raw->hi92.mag_b[2]*0.030517,
             raw->hi92.pitch*0.001, raw->hi92.roll*0.001, raw->hi92.yaw*0.001,
             raw->hi91.quat[0], raw->hi91.quat[1], raw->hi91.quat[2], raw->hi91.quat[3]);
     }
 
     /* dump 0x81 packet */
 else if(raw->hi81.tag == HIPNUC_ID_HI81)
 {
     /* Format:
      * status: device status
      * ins_status: INS algorithm status
      * gpst_wn/tow: GPS week number and time of week
      * gyr: deg/s
      * acc: m/s²
      * mag: uT
      * air_pressure: Pa
      * temperature: °C
      * utc: YYYY-MM-DD HH:mm:ss.SSS
      * pitch/roll/yaw: deg
      * quat: w,x,y,z
      * ins_lat/lon: deg
      * ins_msl: m
      * pdop/hdop: position/horizontal dilution of precision
      * solq_pos: 0:invalid 1:SPP 2:DGPS 4:RTK-FLOAT 5:RTK-FIXED
      * nv_pos: number of satellites used for position
      * solq_heading: 0:invalid 4:valid
      * nv_heading: number of satellites used for heading
      * diff_age: differential age(s)
      * undulation: geoidal separation(m)
      * vel_enu: east,north,up velocity(m/s)
      * acc_enu: east,north,up acceleration(m/s²)
      */
     ret = snprintf(buf + written, buf_size - written,
         "{\n"
         "  \"type\": \"HI81\",\n"
         "  \"status\": %d,\n"
         "  \"ins_status\": %d,\n"
         "  \"gpst_wn\": %d,\n"
         "  \"gpst_tow\": %d,\n"
         "  \"gyr\": [%.3f, %.3f, %.3f],\n"
         "  \"acc\": [%.3f, %.3f, %.3f],\n"
         "  \"mag\": [%.3f, %.3f, %.3f],\n"
         "  \"air_pressure\": %.1f,\n"
         "  \"temperature\": %d,\n"
         "  \"utc\": \"20%02d-%02d-%02d %02d:%02d:%02d.%03d\",\n"
         "  \"pitch\": %.2f,\n"
         "  \"roll\": %.2f,\n"
         "  \"yaw\": %.2f,\n"
         "  \"quat\": [%.3f, %.3f, %.3f, %.3f],\n"
         "  \"ins_lat\": %.7f,\n"
         "  \"ins_lon\": %.7f,\n"
         "  \"ins_msl\": %.2f,\n"
         "  \"pdop\": %.1f,\n"
         "  \"hdop\": %.1f,\n"
         "  \"solq_pos\": %d,\n"
         "  \"nv_pos\": %d,\n"
         "  \"solq_heading\": %d,\n"
         "  \"nv_heading\": %d,\n"
         "  \"diff_age\": %d,\n"
         "  \"undulation\": %.2f,\n"
         "  \"vel_enu\": [%.2f, %.2f, %.2f],\n"
         "  \"acc_enu\": [%.2f, %.2f, %.2f],\n"
         "}\n",
         raw->hi81.status,
         raw->hi81.ins_status,
         raw->hi81.gpst_wn,
         raw->hi81.gpst_tow,
         raw->hi81.gyr_b[0]*(0.001*R2D), raw->hi81.gyr_b[1]*(0.001*R2D), raw->hi81.gyr_b[2]*(0.001*R2D),
         raw->hi81.acc_b[0]*0.0048828, raw->hi81.acc_b[1]*0.0048828, raw->hi81.acc_b[2]*0.0048828,
         raw->hi81.mag_b[0]*0.030517, raw->hi81.mag_b[1]*0.030517, raw->hi81.mag_b[2]*0.030517,
         (float)raw->hi81.air_pressure,
         raw->hi81.temperature,
         raw->hi81.utc_year,
         raw->hi81.utc_month,
         raw->hi81.utc_day,
         raw->hi81.utc_hour,
         raw->hi81.utc_min,
         raw->hi81.utc_msec/1000,
         raw->hi81.utc_msec%1000,
         raw->hi81.pitch*0.01,
         raw->hi81.roll*0.01,
         raw->hi81.yaw*0.01,
         raw->hi81.quat[0]*0.0001, raw->hi81.quat[1]*0.0001, raw->hi81.quat[2]*0.0001, raw->hi81.quat[3]*0.0001,
         raw->hi81.ins_lat*1e-7,
         raw->hi81.ins_lon*1e-7,
         raw->hi81.ins_msl*1e-3,
         raw->hi81.pdop*0.1,
         raw->hi81.hdop*0.1,
         raw->hi81.solq_pos,
         raw->hi81.nv_pos,
         raw->hi81.solq_heading,
         raw->hi81.nv_heading,
         raw->hi81.diff_age,
         raw->hi81.undulation*0.01,
         raw->hi81.vel_enu[0]*0.01, raw->hi81.vel_enu[1]*0.01, raw->hi81.vel_enu[2]*0.01,
         raw->hi81.acc_enu[0]*0.0048828, raw->hi81.acc_enu[1]*0.0048828, raw->hi81.acc_enu[2]*0.0048828);
     }
 
     if (ret > 0) written += ret;
     return written;
 }
 
 /**
  * @brief    Calculate HiPNUC CRC16
  *
  * @param    inital is initial value
  * @param    buf    is input buffer pointer
  * @param    len    is length of the buffer
  */
 static void hipnuc_crc16(uint16_t *inital, const uint8_t *buf, uint32_t len)
 {
     uint32_t crc = *inital;
     uint32_t j;
     for (j=0; j < len; ++j)
     {
         uint32_t i;
         uint32_t byte = buf[j];
         crc ^= byte << 8;
         for (i = 0; i < 8; ++i)
         {
             uint32_t temp = crc << 1;
             if (crc & 0x8000)
             {
                 temp ^= 0x1021;
             }
             crc = temp;
         }
     } 
     *inital = crc;
 }
 