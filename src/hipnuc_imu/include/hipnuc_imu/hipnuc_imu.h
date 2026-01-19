#ifndef HIPNUC_IMU_H
#define HIPNUC_IMU_H

#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"
#include "sensor_msgs/msg/temperature.hpp"
#include "sensor_msgs/msg/fluid_pressure.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

extern "C" {
#include "hipnuc_lib_package/hipnuc_dec.h"
#include "hipnuc_lib_package/nmea_decode.h"
#include "hipnuc_lib_package/hipnuc_can_common.h"
#include "hipnuc_lib_package/hipnuc_j1939_parser.h"
#include "hipnuc_lib_package/canopen_parser.h"
}

#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

#define GRA_ACC     (9.8)
#define DEG_TO_RAD  (0.01745329)

#endif // HIPNUC_IMU_H
