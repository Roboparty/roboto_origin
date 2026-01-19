#include "hipnuc_imu/hipnuc_imu.h"
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>

extern "C" {
#include "hipnuc_lib_package/hipnuc_dec.h"
}

#define BUF_SIZE (1024)

namespace hipnuc_driver {

class SerialDriver
{
public:
    SerialDriver(rclcpp::Node* node) 
        : node_(node), fd_(-1), running_(false)
    {
        write_buffer_ = std::make_shared<sensor_msgs::msg::Imu>();
        read_buffer_ = std::make_shared<sensor_msgs::msg::Imu>();
        memset(&raw_, 0, sizeof(raw_));
    }

    ~SerialDriver()
    {
        stop();
        if (fd_ > 0) {
            close(fd_);
        }
    }

    bool initialize(const std::string& port, int baudrate, const std::string& frame_id)
    {
        write_buffer_->header.frame_id = frame_id;
        read_buffer_->header.frame_id = frame_id;
        
        fd_ = open_serial(port, baudrate);
        if (fd_ < 0) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to open serial port: %s", port.c_str());
            return false;
        }
        
        RCLCPP_INFO(node_->get_logger(), "Serial port opened: %s @ %d baud", port.c_str(), baudrate);
        return true;
    }

    void start()
    {
        if (fd_ < 0) {
            RCLCPP_ERROR(node_->get_logger(), "Cannot start: serial port not initialized");
            return;
        }
        
        running_ = true;
        receive_thread_ = std::thread(&SerialDriver::receiveThread, this);
        RCLCPP_INFO(node_->get_logger(), "Serial receive thread started");
    }

    void stop()
    {
        running_ = false;
        if (receive_thread_.joinable()) {
            receive_thread_.join();
        }
    }

    std::shared_ptr<sensor_msgs::msg::Imu> getData()
    {
        return std::atomic_load(&read_buffer_);
    }

private:
    void receiveThread()
    {
        pthread_setname_np(pthread_self(), "serial_rx");
        
        uint8_t buf[BUF_SIZE] = {0};
        
        while (running_ && rclcpp::ok()) {
            int total_read = 0;
            fd_set readfds;
            struct timeval tv;
            
            while (total_read < BUF_SIZE) {
                FD_ZERO(&readfds);
                FD_SET(fd_, &readfds);
                tv.tv_sec = 0;
                tv.tv_usec = 1000;  // 1ms timeout
                
                int ret = select(fd_ + 1, &readfds, NULL, NULL, &tv);
                if (ret < 0) {
                    if (errno == EINTR) continue;
                    RCLCPP_ERROR(node_->get_logger(), "select error");
                    return;
                } else if (ret == 0) {
                    break;  // timeout
                }
                
                ret = read(fd_, buf + total_read, BUF_SIZE - total_read);
                if (ret < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
                    RCLCPP_ERROR(node_->get_logger(), "read error");
                    return;
                } else if (ret == 0) {
                    break;
                }
                
                total_read += ret;
            }
            
            if (total_read > 0) {
                for (int i = 0; i < total_read; i++) {
                    if (hipnuc_input(&raw_, buf[i])) {
                        write_buffer_->orientation.w = raw_.hi91.quat[0];
                        write_buffer_->orientation.x = raw_.hi91.quat[1];
                        write_buffer_->orientation.y = raw_.hi91.quat[2];
                        write_buffer_->orientation.z = raw_.hi91.quat[3];
                        
                        write_buffer_->angular_velocity.x = raw_.hi91.gyr[0] * DEG_TO_RAD;
                        write_buffer_->angular_velocity.y = raw_.hi91.gyr[1] * DEG_TO_RAD;
                        write_buffer_->angular_velocity.z = raw_.hi91.gyr[2] * DEG_TO_RAD;
                        
                        write_buffer_->linear_acceleration.x = raw_.hi91.acc[0] * GRA_ACC;
                        write_buffer_->linear_acceleration.y = raw_.hi91.acc[1] * GRA_ACC;
                        write_buffer_->linear_acceleration.z = raw_.hi91.acc[2] * GRA_ACC;
                        
                        write_buffer_->header.stamp = node_->now();
                        
                        write_buffer_ = std::atomic_exchange(&read_buffer_, write_buffer_);
                    }
                }
                memset(buf, 0, sizeof(buf));
            }
        }
    }

    int open_serial(const std::string& port, int baud)
    {
        int fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd < 0) {
            return -1;
        }

        struct termios tty;
        memset(&tty, 0, sizeof(tty));
        
        if (tcgetattr(fd, &tty) != 0) {
            close(fd);
            return -1;
        }

        #ifdef __linux__
        #include <asm/termbits.h>
        struct termios2 tty2;
        if (ioctl(fd, TCGETS2, &tty2) == 0) {
            tty2.c_cflag &= ~CBAUD;
            tty2.c_cflag |= BOTHER;
            tty2.c_ispeed = baud;
            tty2.c_ospeed = baud;
            tty2.c_cflag &= ~CSIZE;
            tty2.c_cflag |= CS8;
            tty2.c_cflag &= ~PARENB;
            tty2.c_cflag &= ~CSTOPB;
            tty2.c_cflag |= (CLOCAL | CREAD);
            tty2.c_cflag &= ~CRTSCTS;
            tty2.c_lflag &= ~(ICANON | ECHO | ECHOE | ECHONL | ISIG);
            tty2.c_iflag &= ~(IXON | IXOFF | IXANY);
            tty2.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
            tty2.c_oflag &= ~OPOST;
            tty2.c_cc[VMIN] = 0;
            tty2.c_cc[VTIME] = 0;
            
            if (ioctl(fd, TCSETS2, &tty2) == 0) {
                tcflush(fd, TCIOFLUSH);
                return fd;
            }
        }
        #endif

        speed_t speed;
        switch (baud) {
            case 9600: speed = B9600; break;
            case 19200: speed = B19200; break;
            case 38400: speed = B38400; break;
            case 57600: speed = B57600; break;
            case 115200: speed = B115200; break;
            case 230400: speed = B230400; break;
            case 460800: speed = B460800; break;
            case 921600: speed = B921600; break;
            default:
                close(fd);
                return -1;
        }

        cfsetispeed(&tty, speed);
        cfsetospeed(&tty, speed);
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag |= (CLOCAL | CREAD);
        tty.c_cflag &= ~CRTSCTS;
        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ECHONL | ISIG);
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
        tty.c_oflag &= ~OPOST;
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 0;

        if (tcsetattr(fd, TCSANOW, &tty) != 0) {
            close(fd);
            return -1;
        }

        tcflush(fd, TCIOFLUSH);
        return fd;
    }

    rclcpp::Node* node_;
    int fd_;
    std::atomic<bool> running_;
    std::thread receive_thread_;
    
    hipnuc_raw_t raw_;
    std::shared_ptr<sensor_msgs::msg::Imu> write_buffer_;
    std::shared_ptr<sensor_msgs::msg::Imu> read_buffer_;
};

} // namespace hipnuc_driver
