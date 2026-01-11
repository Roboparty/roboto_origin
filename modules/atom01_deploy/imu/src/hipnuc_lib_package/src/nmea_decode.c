#include "hipnuc_lib_package/nmea_decode.h"

#define MAXFIELD 64 /* max number of fields in a record */

/* sync code */
static int sync_nmea(uint8_t *buf, uint8_t data)
{
    buf[0] = buf[1];
    buf[1] = data;
    return buf[0] == '$' && buf[1] == 'G';
}

static double dmm2deg(double dmm)
{
    return floor(dmm / 100.0) + fmod(dmm, 100.0) / 60.0;
}

static void septime(double t, double *t1, double *t2, double *t3)
{
    *t1 = floor(t / 10000.0);
    t -= *t1 * 10000.0;
    *t2 = floor(t / 100.0);
    *t3 = t - *t2 * 100.0;
}

/* decode NMEA GGA (Global Positioning System Fix Data) sentence -------------*/
static int dec_gga(nmea_raw_t *raw, char **val, int n)
{
    double tod = 0.0, lat = 0.0, lon = 0.0, hdop = 0.0, alt = 0.0, msl = 0.0;
    double age = 0.0;
    char ns = 'N', ew = 'E', ua = ' ', um = ' ';
    int i, solq = 0, nrcv = 0;

    for (i = 0; i < n; i++)
    {
        switch (i)
        {
        case 0:
            tod = atof(val[i]);
            break; /* UTC of position (hhmmss) */
        case 1:
            lat = atof(val[i]);
            break; /* Latitude (ddmm.mmm) */
        case 2:
            ns = *val[i];
            break; /* N=north,S=south */
        case 3:
            lon = atof(val[i]);
            break; /* Longitude (dddmm.mmm) */
        case 4:
            ew = *val[i];
            break; /* E=east,W=west */
        case 5:
            solq = atof(val[i]);
            break; /* GPS quality indicator */
        case 6:
            nrcv = atof(val[i]);
            break; /* # of satellites in use */
        case 7:
            hdop = atof(val[i]);
            break; /* HDOP */
        case 8:
            alt = atof(val[i]);
            break; /* Altitude MSL */
        case 9:
            ua = *val[i];
            break; /* unit (M) */
        case 10:
            msl = atof(val[i]);
            break; /* Geoid separation */
        case 11:
            um = *val[i];
            break; /* unit (M) */
        case 12:
            age = atof(val[i]);
            break; /* Age of differential */
        default:
            break;
        }
    }
    if ((ns != 'N' && ns != 'S') || (ew != 'E' && ew != 'W'))
    {
        // NL_TRACE("invalid nmea gga format\r\n");
        return 0;
    }

    raw->gga.lat = (ns == 'N' ? 1.0 : -1.0) * dmm2deg(lat);
    raw->gga.lon = (ew == 'E' ? 1.0 : -1.0) * dmm2deg(lon);
    raw->gga.alt = alt;
    raw->gga.undulation = msl;
    raw->gga.nv = nrcv;
    raw->gga.status = solq;
    raw->gga.hdop = hdop;

    septime(tod, &raw->gga.hour, &raw->gga.min, &raw->gga.sec);
    return 1;
}

static int dec_rmc(nmea_raw_t *raw, char **val, int n)
{
    volatile double tod = 0.0, lat = 0.0, lon = 0.0, vel = 0.0, dir = 0.0, date = 0.0, ang = 0.0;
    char act = ' ', ns = 'N', ew = 'E', mew = 'E', mode = 'A';
    int i;

    for (i = 0; i < n; i++)
    {
        switch (i)
        {
        case 0:
            tod = atof(val[i]);
            break; /* time in utc (hhmmss) */
        case 1:
            act = *val[i];
            break; /* A=active,V=void */
        case 2:
            lat = atof(val[i]);
            break; /* latitude (ddmm.mmm) */
        case 3:
            ns = *val[i];
            break; /* N=north,S=south */
        case 4:
            lon = atof(val[i]);
            break; /* longitude (dddmm.mmm) */
        case 5:
            ew = *val[i];
            break; /* E=east,W=west */
        case 6:
            vel = atof(val[i]);
            break; /* speed (knots) */
        case 7:
            dir = atof(val[i]);
            break; /* track angle (deg) */
        case 8:
            date = atof(val[i]);
            break; /* date (ddmmyy) */
        case 9:
            ang = atof(val[i]);
            break; /* magnetic variation */
        case 10:
            mew = *val[i];
            break; /* E=east,W=west */
        case 11:
            mode = *val[i];
            break; /* mode indicator (>nmea 2) */
                   /* A=autonomous,D=differential */
                   /* E=estimated,N=not valid,S=simulator */
        }
    }
    if ((act != 'A' && act != 'V') || (ns != 'N' && ns != 'S') || (ew != 'E' && ew != 'W'))
    {
        // NL_TRACE("invalid nmea rmc format\n");
        return 0;
    }

    raw->rmc.lat = (ns == 'N' ? 1.0 : -1.0) * dmm2deg(lat);
    raw->rmc.lon = (ew == 'E' ? 1.0 : -1.0) * dmm2deg(lon);
    raw->rmc.mode = mode;
    raw->rmc.sog = vel;
    raw->rmc.cog = dir;

    septime(date, &raw->rmc.day, &raw->rmc.mouth, &raw->rmc.year);
    septime(tod, &raw->rmc.hour, &raw->rmc.min, &raw->rmc.sec);
    raw->rmc.year += raw->rmc.year < 80.0 ? 2000.0 : 1900.0;
    return 1;
}

static int dec_gsa(nmea_raw_t *raw, char **val, int n)
{
    int i, id;
    raw->gsa.mode = *val[0];
    sscanf(val[1], "%d", (int *)&raw->gsa.sol_mode);
    sscanf(val[14], "%f", &raw->gsa.pdop);
    sscanf(val[15], "%f", &raw->gsa.hdop);
    sscanf(val[16], "%f", &raw->gsa.vdop);
    sscanf(val[17], "%d", (int *)&raw->gsa.sys);

    for (i = 2, raw->gsa.nv = 0; i < 13; i++)
    {
        if (sscanf(val[i], "%d", &id) == 1)
            raw->gsa.sat[raw->gsa.nv++] = id;
    }

    // printf("sol_mode:%d, PDOP:%.1f HDOP:%.1f VDOP:%.1f, YS:%d SAT:", raw->gsa.sol_mode, raw->gsa.pdop, raw->gsa.hdop, raw->gsa.vdop, raw->gsa.sys);
    // for(i=0; i<raw->gsa.nv; i++)
    //{
    //   printf("%d,", raw->gsa.sat[i]);
    // }
    // printf("\r\n");

    return 1;
}

static int dec_gsv(nmea_raw_t *raw, char **val, int n)
{
    static uint8_t sat_ctr = 0;

    int i;
    sscanf(val[0], "%d", (int *)&raw->gsv.n_sentences);
    sscanf(val[1], "%d", (int *)&raw->gsv.idx_sentences);
    sscanf(val[2], "%d", (int *)&raw->gsv.nv);
    sscanf(val[n - 1], "%x", (int *)&raw->gsv.sid);

    switch (raw->buf[2])
    {
    case 'P':
        raw->gsv.sys = 1;
        break;
    case 'L':
        raw->gsv.sys = 2;
        break;
    case 'A':
        raw->gsv.sys = 3;
        break;
    case 'B':
        raw->gsv.sys = 4;
        break;
    case 'Q':
        raw->gsv.sys = 5;
        break;
    }

    if (raw->gsv.idx_sentences == 1)
        sat_ctr = 0;

    for (i = 3; i < n - 4; i += 4)
    {
        sscanf(val[i], "%02d", (int *)&(raw->gsv.sat[sat_ctr].sat_id));
        sscanf(val[i + 1], "%02d", (int *)&raw->gsv.sat[sat_ctr].el);
        sscanf(val[i + 2], "%03d", (int *)&raw->gsv.sat[sat_ctr].az);
        sscanf(val[i + 3], "%02d", (int *)&raw->gsv.sat[sat_ctr++].snr);
    }

    // if(raw->gsv.n_sentences == raw->gsv.idx_sentences)
    //{
    //     printf("nv:%d, sys:%d,singal:%d,", raw->gsv.nv, raw->gsv.sys, raw->gsv.sid);
    //     printf("SATINFO:\r\n");
    //     for(i=0; i<sat_ctr; i++)
    //     {
    //       printf("[%02d:%d %d %d],", raw->gsv.sat[i].sat_id, raw->gsv.sat[i].el, raw->gsv.sat[i].az, raw->gsv.sat[i].snr);
    //     }
    //     printf("\r\n");
    // }

    return (raw->gsv.n_sentences == raw->gsv.idx_sentences) ? (1) : (0);
}

static int dec_sxt(nmea_raw_t *raw, char **val, int n)
{
    volatile double date =0.0, lon =0.0, lat =0.0, alt =0.0,  yaw =0.0, pitch =0.0, dir =0.0, vel =0.0, roll =0.0, gyr_z =0.0,
            e_v =0.0, n_v =0.0, u_v =0.0;
    char solq =0 , solq_heading =0, nv =0, nv_heading =0;
    int i;

    for (i = 0; i < n; i++)
    {
        switch (i)
        {
        case 0:
            date = atof(val[i]);
            break; /* time in utc (hhmmss) */
        case 1:
            lon = atof(val[i]);
            break; /* longitude (ddd.dddddddd) */
        case 2:
            lat = atof(val[i]);
            break; /* latitude (ddd.dddddddd) */
        case 3:
            alt = atof(val[i]);
            break; /* altitude (ddd.dddd ) */
        case 4:
            yaw = atof(val[i]);
            break; /* yaw (ddd.dd) */
        case 5:
            pitch = atof(val[i]);
            break; /* pitch (ddd.dd) */
        case 6:
            dir = atof(val[i]);
            break; /* speed angle (deg) */
        case 7:
            vel = atof(val[i]);
            break; /* speed (knots) */
        case 8:
            roll = atof(val[i]);
            break; /* roll (ddd.dd) */
        case 9:
            solq =atoi(val[i]);
            break; /* status (0 1 2 3) */
        case 10:
            solq_heading =atoi(val[i]);
            break; /* status (0 1 2 3) */
        case 11:
            nv_heading =atoi(val[i]);
            break; /* slave satellite number */
        case 12:
            nv =atoi(val[i]);
            break; /* smain satellite number */

        case 15:
            gyr_z = atof(val[i]);
            break; /* z gyr (ddmmyy) */
        case 16:
            e_v = atof(val[i]);
            break; /* enu e speed */
        case 17:
            n_v = atof(val[i]);
            break; /* enu n speed */
        case 18:
            u_v = atof(val[i]);
            break; /* enu u speed */
        case 19:
            raw->sxt.ins_stat = atoi(val[i]);
            break; /* enu u speed */
        }
    }

    raw->sxt.year = date/10000000000;
    raw->sxt.month = (date -raw->sxt.year*10000000000)/100000000;
    raw->sxt.day = (date -raw->sxt.year*10000000000 -raw->sxt.month*100000000)/1000000;
    septime(date -date/1000000 *1000000, &raw->sxt.hour, &raw->sxt.min, &raw->sxt.sec);
    raw->sxt.lon = lon;
    raw->sxt.lat = lat;
    raw->sxt.alt = alt;
    raw->sxt.yaw = yaw;
    raw->sxt.pitch = pitch;
    raw->sxt.dir = dir;
    raw->sxt.vel = vel;
    raw->sxt.solq = solq;
    raw->sxt.solq_heading = solq_heading;
    raw->sxt.nv_heading =nv_heading;
    raw->sxt.nv = nv;
    raw->sxt.roll = roll;
    raw->sxt.gyr_z = gyr_z;
    raw->sxt.e_v = e_v;
    raw->sxt.n_v = n_v;
    raw->sxt.u_v = u_v;

    return 1;
}

static int parse_nmea(nmea_raw_t *raw)
{
    char *p, *q, *val[MAXFIELD], sum, sum2;
    int n = 0;

    /* checksum */
    for (q = (char *)raw->buf + 1, sum = 0; *q && *q != '*'; q++)
        sum ^= *q;
    q++;
    sum2 = (int)strtol(q, NULL, 16);

    if (sum != sum2)
    {
        // NL_TRACE("nmea checksum failed %02X %02X\n",sum, sum2);
        return 0;
    }

    /* parse fields */
    for (p = (char *)raw->buf; *p && n < MAXFIELD; p = q + 1)
    {
        if ((q = strchr(p, ',')) != NULL || (q = strchr(p, '*')) != NULL)
        {
            val[n++] = p;
            *q = '\0';
        }
        else
            break;
    }
    if (n < 1)
    {
        return 0;
    }

    /* copy NMEA header */
    memcpy(raw->type, raw->buf + 3, 3);

    if (!strncmp(val[0] + 3, "GGA", 3))
    {
        return dec_gga(raw, val + 1, n - 1);
    }
    if (!strncmp(val[0] + 3, "RMC", 3))
    {
        return dec_rmc(raw, val + 1, n - 1);
    }
    if (!strncmp(val[0] + 3, "GSV", 3))
    {
        return dec_gsv(raw, val + 1, n - 1);
    }
    if (!strncmp(val[0] + 3, "GSA", 3))
    {
        return dec_gsa(raw, val + 1, n - 1);
    }
    if (!strncmp(val[0] + 3, "SXT", 3))
    {
        return dec_sxt(raw, val + 1, n - 1);
    }

    return 0;
}

int input_nmea(nmea_raw_t *raw, uint8_t data)
{
    /* synchronize frame */
    if (raw->nbyte == 0)
    {
        if (!sync_nmea(raw->buf, data))
            return 0;
        raw->nbyte = 2;
        return 0;
    }

    if (raw->nbyte >= MAXRAWLEN)
    {
        //        NL_TRACE("ch length error: len=%d\n",raw->nbyte);
        raw->nbyte = 0;
        return -1;
    }

    raw->buf[raw->nbyte++] = data;

    if (data != 0x0A)
    {
        return 0;
    }

    raw->len = raw->nbyte;
    raw->nbyte = 0;
    // return 1;
   return parse_nmea(raw);
}
