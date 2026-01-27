#include "serial_port.hpp"

std::shared_ptr<spdlog::logger> SerialPort::logger_ = nullptr;

SerialPort::SerialPort(const std::string& interface, int baudrate) 
    : interface_(interface), baudrate_(baudrate), fd_(-1), running_(false) {
    init();
}

void SerialPort::init() {
    fd_ = ::open(interface_.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd_ < 0) {
        if (logger_) logger_->error("Failed to open serial port: {}", interface_);
        else std::cerr << "Failed to open serial port: " << interface_ << std::endl;
        throw std::runtime_error("Failed to open serial port: " + interface_);
    }

    struct termios tty;
    if (tcgetattr(fd_, &tty) != 0) {
        if (logger_) logger_->error("Failed to get serial attributes: {}", interface_);
        else std::cerr << "Failed to get serial attributes: " << interface_ << std::endl;
        ::close(fd_);
        throw std::runtime_error("Failed to get serial attributes: " + interface_);
    }

    bool configured = false;

    if (!configured) {
        speed_t speed;
        switch (baudrate_) {
            case 9600: speed = B9600; break;
            case 19200: speed = B19200; break;
            case 38400: speed = B38400; break;
            case 57600: speed = B57600; break;
            case 115200: speed = B115200; break;
            case 230400: speed = B230400; break;
            case 460800: speed = B460800; break;
            case 921600: speed = B921600; break;
            default: speed = B115200; break; 
        }

        cfsetispeed(&tty, speed);
        cfsetospeed(&tty, speed);

        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag |= (CLOCAL | CREAD);
        tty.c_cflag &= ~CRTSCTS;

        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
        tty.c_oflag &= ~OPOST;

        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 1;

        if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
            if (logger_) logger_->error("Failed to set serial attributes: {}", interface_);
            else std::cerr << "Failed to set serial attributes: " << interface_ << std::endl;
            ::close(fd_);
            throw std::runtime_error("Failed to set serial attributes: " + interface_);
        }
    }

    running_ = true;
    rx_thread_ = std::thread([this]() {
        pthread_setname_np(pthread_self(), "serial_rx");
        struct sched_param sp{}; sp.sched_priority = 80;
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
            
        uint8_t buf[BUF_SIZE] = {0};
        
        while (running_) {
            fd_set readfds;
            struct timeval tv;
            
            FD_ZERO(&readfds);
            FD_SET(fd_, &readfds);
            tv.tv_sec = 0;
            tv.tv_usec = 1000;  // 1ms timeout

            int ret = select(fd_ + 1, &readfds, NULL, NULL, &tv);
            if (ret < 0) {
                if (errno == EINTR) continue;
                if (logger_) logger_->error("select error: {}", strerror(errno));
                else std::cerr << "select error: " << strerror(errno) << std::endl;
                break;
            } else if (ret == 0) {
                continue;  // timeout
            }
            
            int n = read(fd_, buf, BUF_SIZE);
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
                if (logger_) logger_->error("read error: {}", strerror(errno));
                else std::cerr << "read error: " << strerror(errno) << std::endl;
                break; 
            } else if (n > 0) {
                if (callback_) {
                    callback_(buf, n);
                }
            }
        }
    });
}

SerialPort::~SerialPort() {
    close();
}

std::shared_ptr<SerialPort> SerialPort::open(const std::string& interface, int baudrate) {
    return std::shared_ptr<SerialPort>(new SerialPort(interface, baudrate));
}

void SerialPort::close() {
    running_ = false;
    if (rx_thread_.joinable()) {
        rx_thread_.join();
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void SerialPort::set_serial_callback(SerialCbkFunc callback) {
    callback_ = callback;
}

