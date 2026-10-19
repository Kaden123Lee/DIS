#pragma once

#include <sys/socket.h>
#include <netinet/in.h>
#include <cerrno>
#include <cstdint>
#include <string>

// Both peers use a 32-bit network-order length followed by the message bytes.
constexpr std::uint32_t max_message_size = 16 * 1024 * 1024;

inline bool recv_all(int fd, void* data, std::size_t size) {
    auto* bytes = static_cast<char*>(data);
    while (size > 0) {
        ssize_t count = recv(fd, bytes, size, 0);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        bytes += count;
        size -= static_cast<std::size_t>(count);
    }
    return true;
}

inline bool send_all(int fd, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const char*>(data);
    while (size > 0) {
        ssize_t count = send(fd, bytes, size, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        bytes += count;
        size -= static_cast<std::size_t>(count);
    }
    return true;
}

inline int send_message(int fd, const std::string& message) {
    if (message.size() > max_message_size) {
        errno = EMSGSIZE;
        return -1;
    }
    std::uint32_t size = htonl(static_cast<std::uint32_t>(message.size()));
    return send_all(fd, &size, sizeof(size)) &&
           send_all(fd, message.data(), message.size()) ? 0 : -1;
}

inline std::string receive_message(int fd) {
    std::uint32_t size;
    if (!recv_all(fd, &size, sizeof(size))) return "";
    size = ntohl(size);
    if (size > max_message_size) {
        errno = EMSGSIZE;
        return "";
    }
    std::string message(size, '\0');
    if (!recv_all(fd, message.data(), size)) return "";
    return message;
}
