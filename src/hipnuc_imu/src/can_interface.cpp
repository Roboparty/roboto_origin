#include "hipnuc_imu/hipnuc_imu.h"
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>

extern "C" {
#include "hipnuc_lib_package/hipnuc_can_common.h"
#include "hipnuc_lib_package/hipnuc_j1939_parser.h"
#include "hipnuc_lib_package/canopen_parser.h"
}

namespace hipnuc_driver {

class CANDriver
{
public:
    CANDriver(rclcpp::Node* node)
        : node_(node), sockfd_(-1), running_(false)
    {
        memset(&sensor_data_, 0, sizeof(sensor_data_));
    }

    ~CANDriver()
    {
        stop();
        if (sockfd_ >= 0) {
            close(sockfd_);
        }
    }

    bool initialize(const std::string& interface, const std::string& frame_id)
    {
        frame_id_ = frame_id;
        
        sockfd_ = can_open_socket_internal(interface.c_str());
        if (sockfd_ < 0) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to open CAN interface: %s", interface.c_str());
            return false;
        }
        
        RCLCPP_INFO(node_->get_logger(), "CAN interface opened: %s (socket: %d)", interface.c_str(), sockfd_);
        return true;
    }

    void start()
    {
        if (sockfd_ < 0) {
            RCLCPP_ERROR(node_->get_logger(), "Cannot start: CAN socket not initialized");
            return;
        }
        
        running_ = true;
        receive_thread_ = std::thread(&CANDriver::receiveThread, this);
        RCLCPP_INFO(node_->get_logger(), "CAN receive thread started");
    }

    void stop()
    {
        running_ = false;
        if (receive_thread_.joinable()) {
            receive_thread_.join();
        }
    }

    bool getIMUData(sensor_msgs::msg::Imu& msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        msg.header.frame_id = frame_id_;
        msg.header.stamp = node_->now();
        
        msg.orientation.w = sensor_data_.quat_w;
        msg.orientation.x = sensor_data_.quat_x;
        msg.orientation.y = sensor_data_.quat_y;
        msg.orientation.z = sensor_data_.quat_z;
        
        msg.angular_velocity.x = sensor_data_.gyr_x * DEG_TO_RAD;
        msg.angular_velocity.y = sensor_data_.gyr_y * DEG_TO_RAD;
        msg.angular_velocity.z = sensor_data_.gyr_z * DEG_TO_RAD;
        
        msg.linear_acceleration.x = sensor_data_.acc_x * GRA_ACC;
        msg.linear_acceleration.y = sensor_data_.acc_y * GRA_ACC;
        msg.linear_acceleration.z = sensor_data_.acc_z * GRA_ACC;
        
        return true;
    }

    bool getMagData(sensor_msgs::msg::MagneticField& msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        msg.header.frame_id = frame_id_;
        msg.header.stamp = node_->now();
        msg.magnetic_field.x = sensor_data_.mag_x;
        msg.magnetic_field.y = sensor_data_.mag_y;
        msg.magnetic_field.z = sensor_data_.mag_z;
        
        return true;
    }

    bool getEulerData(geometry_msgs::msg::Vector3Stamped& msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        msg.header.frame_id = frame_id_;
        msg.header.stamp = node_->now();
        msg.vector.x = sensor_data_.roll;
        msg.vector.y = sensor_data_.pitch;
        msg.vector.z = sensor_data_.imu_yaw;
        
        return true;
    }

    bool getTemperatureData(sensor_msgs::msg::Temperature& msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        msg.header.frame_id = frame_id_;
        msg.header.stamp = node_->now();
        msg.temperature = sensor_data_.temperature;
        
        return true;
    }

    bool getPressureData(sensor_msgs::msg::FluidPressure& msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        msg.header.frame_id = frame_id_;
        msg.header.stamp = node_->now();
        msg.fluid_pressure = sensor_data_.pressure;
        
        return true;
    }

    bool getGPSData(sensor_msgs::msg::NavSatFix& msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        msg.header.frame_id = frame_id_;
        msg.header.stamp = node_->now();
        msg.latitude = sensor_data_.ins_lat;
        msg.longitude = sensor_data_.ins_lon;
        msg.altitude = sensor_data_.ins_msl;
        
        return true;
    }

    bool getVelocityData(geometry_msgs::msg::TwistStamped& msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        msg.header.frame_id = frame_id_;
        msg.header.stamp = node_->now();
        msg.twist.linear.x = sensor_data_.ins_vel_n;
        msg.twist.linear.y = sensor_data_.ins_vel_e;
        msg.twist.linear.z = sensor_data_.ins_vel_u;
        
        return true;
    }

private:
    void receiveThread()
    {
        pthread_setname_np(pthread_self(), "can_rx");
        
        struct can_frame linux_frame;
        struct timespec ts;
        
        RCLCPP_INFO(node_->get_logger(), "CAN receive thread running, waiting for frames...");
        int frame_count = 0;
        
        while (running_ && rclcpp::ok()) {
            int nbytes = can_receive_frame_ts_internal(sockfd_, &linux_frame, &ts);
            if (nbytes < 0) {
                if (errno == EINTR) continue;
                RCLCPP_ERROR(node_->get_logger(), "CAN receive error: %s", strerror(errno));
                break;
            }
            
            if (nbytes == 0) continue;
            
            frame_count++;
            if (frame_count % 100 == 0) {
                RCLCPP_INFO(node_->get_logger(), "Received %d CAN frames (last ID: 0x%X)", 
                            frame_count, linux_frame.can_id);
            }
            
            hipnuc_can_frame_t frame;
            frame.can_id = linux_frame.can_id;
            frame.can_dlc = linux_frame.can_dlc;
            memcpy(frame.data, linux_frame.data, 8);
            frame.hw_ts_us = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000ULL;
            
            std::lock_guard<std::mutex> lock(data_mutex_);
            
            int ret = hipnuc_j1939_parse_frame(&frame, &sensor_data_);
            if (ret == 0) {
                continue;
            }
            
            ret = canopen_parse_frame(&frame, &sensor_data_);
            if (ret == 0) {
                continue;
            }
        }
        
        RCLCPP_INFO(node_->get_logger(), "CAN receive thread stopped. Total frames: %d", frame_count);
    }

    int can_open_socket_internal(const char* ifname)
    {
        int s = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (s < 0) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to create CAN socket");
            return -1;
        }

        struct ifreq ifr;
        strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
        ifr.ifr_name[IFNAMSIZ - 1] = '\0';

        if (ioctl(s, SIOCGIFINDEX, &ifr) < 0) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to get interface index for %s", ifname);
            close(s);
            return -1;
        }

        struct sockaddr_can addr;
        memset(&addr, 0, sizeof(addr));
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;

        if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to bind CAN socket");
            close(s);
            return -1;
        }

        int enable = 1;
        setsockopt(s, SOL_SOCKET, SO_TIMESTAMP, &enable, sizeof(enable));

        return s;
    }

    int can_receive_frame_ts_internal(int sockfd, struct can_frame* frame, struct timespec* ts)
    {
        struct iovec iov;
        struct msghdr msg;
        char ctrlmsg[CMSG_SPACE(sizeof(struct timeval))];

        iov.iov_base = frame;
        iov.iov_len = sizeof(*frame);

        memset(&msg, 0, sizeof(msg));
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = ctrlmsg;
        msg.msg_controllen = sizeof(ctrlmsg);

        int nbytes = recvmsg(sockfd, &msg, 0);
        if (nbytes < 0) {
            return -1;
        }

        struct cmsghdr* cmsg;
        for (cmsg = CMSG_FIRSTHDR(&msg); cmsg != NULL; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SO_TIMESTAMP) {
                struct timeval* tv = (struct timeval*)CMSG_DATA(cmsg);
                ts->tv_sec = tv->tv_sec;
                ts->tv_nsec = tv->tv_usec * 1000;
                break;
            }
        }

        return nbytes;
    }

    rclcpp::Node* node_;
    int sockfd_;
    std::atomic<bool> running_;
    std::thread receive_thread_;
    std::string frame_id_;
    
    std::mutex data_mutex_;
    can_sensor_data_t sensor_data_;
};

} // namespace hipnuc_driver
