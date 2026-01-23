/**
 * @file
 * This file declares an interface to SocketCAN,
 * to facilitates frame transmission and reception.
 */

#pragma once

#include <linux/can.h>
#include <net/if.h>
#include <pthread.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>

#include <atomic>
#include <boost/lockfree/queue.hpp>
#include <condition_variable>
#include <cstdbool>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

constexpr const int INIT_FD = -1;
constexpr const int TIMEOUT_SEC = 0;
constexpr const int TIMEOUT_USEC = 1000;
constexpr const int TX_QUEUE_SIZE = 4096;
constexpr const int MAX_RETRY_COUNT = 3;

using LFQueue = boost::lockfree::queue<can_frame, boost::lockfree::fixed_sized<true>>;
using CanCbkFunc = std::function<void(const can_frame &)>;
using CanCbkId = uint16_t;
using CanCbkMap = std::unordered_map<CanCbkId, CanCbkFunc>;
using CanCbkKeyExtractor = std::function<CanCbkId(const can_frame &)>;

class SocketCAN {
   private:
    std::string interface_;  // The network interface name
    int sockfd_ = -1;        // The file descriptor for the CAN socket
    std::atomic<bool> receiving_;
    LFQueue tx_queue_;
    std::mutex tx_mutex_;
    std::condition_variable tx_cv_;

    sockaddr_can addr_;      // The address of the CAN socket
    ifreq if_request_;       // The network interface request

    /// Receiving
    std::thread receiver_thread_;
    CanCbkMap can_callback_list_;
    std::mutex can_callback_mutex_;
    CanCbkKeyExtractor key_extractor_ = [](const can_frame &frame) -> CanCbkId {
        return static_cast<CanCbkId>(frame.can_id);
    };

    /// Transmitting
    std::thread sender_thread_;

    SocketCAN(std::string interface);

    static std::shared_ptr<SocketCAN> createInstance(const std::string &interface) {
        return std::shared_ptr<SocketCAN>(new SocketCAN(interface));
    }
    static std::shared_ptr<spdlog::logger> logger_;
    static std::unordered_map<std::string, std::shared_ptr<SocketCAN>> instances_;

   public:
    SocketCAN(const SocketCAN &) = delete;
    SocketCAN &operator=(const SocketCAN &) = delete;
    ~SocketCAN();
    static void init_logger(std::shared_ptr<spdlog::logger> logger) { logger_ = logger; }
    static std::shared_ptr<SocketCAN> get(std::string interface) {
        if (logger_.get() == nullptr) logger_ = spdlog::stdout_color_mt("SocketCAN");
        if (instances_.find(interface) == instances_.end()) instances_[interface] = createInstance(interface);
        return instances_[interface];
    }
    void open(std::string interface);
    void close();
    void transmit(const can_frame &frame);
    void add_can_callback(const CanCbkFunc callback, const CanCbkId id);
    void remove_can_callback(const CanCbkId id);
    void clear_can_callbacks();
    void set_key_extractor(CanCbkKeyExtractor extractor);
};
