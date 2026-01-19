#ifndef _DECODE_NMEA_H
#define _DECODE_NMEA_H

#include <stdio.h>
#include <stdlib.h>
#include "stdint.h"
#include <string.h>
#include <math.h>

#define MAXRAWLEN   1800

typedef struct
{
    double        hour;
    double        min;
    double        sec;
    double        lat,lon,alt;  /* lat: in deg, lon: in deg, alt:m(msl) */
    uint8_t       nv;
    uint8_t       status;
    float         hdop;
    float         undulation;
    float         diff_age;
    uint32_t      sta_id;
}nmea_gga_t;

typedef struct
{
    double       year;
    double       mouth;
    double       day;
    double       hour;
    double       min;
    double       sec;
    uint8_t      mode;
    double       lat,lon;
    float        sog; /* speed over ground, knots */
    float        cog; /* Course over ground, deg */
}nmea_rmc_t;


typedef struct
{
  uint16_t sat_id; /* satellite ID number */
  uint16_t el;     /* Elevation, degrees, */
  uint16_t az;     /* Azimuth, degrees */
  uint16_t snr;    /* SNR (C/No) 00-99 dB-Hz, null when not tracking */
}gsv_sat_t;

typedef struct
{
  uint8_t      n_sentences;       /* total number of sentences */
  uint8_t      idx_sentences;     /* sentence number */
  uint8_t      nv;                /*  total number of satellites in view */
  gsv_sat_t    sat[32];            /* sat information */
  uint8_t      sid;               /* signal id see: Table 21 - GNSS Identification Table - GSV*/
  uint8_t      sys;
}nmea_gsv_t;



typedef struct
{
  char mode;        /* M = Manual, forced to operate in 2D or 3D mode A = Automatic, allowed to automatically switch 2D/3D */
  uint8_t sol_mode; /* 1 = Fix not available, 2 = 2D, 3 = 3D */
  uint8_t sat[16];  /* valid sat array, see GSA doc */
  uint8_t nv;       /* # of valid sat in sat[] */
  float pdop;
  float hdop;
  float vdop;
  uint8_t sys;     /* 1:GP  2:GL  3:GA  4:BD(GB) 5:QZSS */
}nmea_gsa_t;

typedef struct
{
    double year;
    double month;
    double day;
    double hour;
    double min;
    double sec;
    double lon;
    double lat;
    double alt;
    double yaw;
    double pitch;
    double dir;
    double vel;
    uint8_t solq;         /* align with GGA's indicator */
    uint8_t solq_heading; /* aligne with GGA's indicator */
    uint8_t nv_heading;
    uint8_t nv;
    double roll;
    double gyr_z;
    double e_v;
    double n_v;
    double u_v;
    uint8_t ins_stat;   /* aligned with ins->ins_stat */
}nmea_sxt_t;

typedef struct
{
    int         nbyte;                  /* number of bytes in message buffer */ 
    int         len;
    uint8_t     buf[MAXRAWLEN];         /* message raw buffer */
    char type[3];                     /* which type of nmea sentense "GGA", "RMC", "GSV", "GSA" */
    nmea_gga_t gga;
    nmea_rmc_t rmc;
    nmea_gsa_t gsa;
    nmea_gsv_t gsv;
    nmea_sxt_t sxt;
}nmea_raw_t;

int input_nmea(nmea_raw_t *raw, uint8_t data);

#endif
