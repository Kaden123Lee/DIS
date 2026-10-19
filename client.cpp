#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <cstdint>
#include <limits>
#include "socket_messages.h"
#include <string>

int start_client() {
    int client_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd == -1) {
        perror("socket");
        return -1;
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(8080);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (connect(client_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == -1) {
        perror("connect");
        close(client_fd);
        return -1;
    }

    return client_fd;
}

// !llama ---------------------------------------------------------------------------
/*
| Function | Purpose | When called |
|---|---|---|
| `send_prompt()` | Send a prompt and wait for the model's response | Every request |



*/
std::string send_prompt(int socket_fd, const std::string& prompt) {
    if (send_message(socket_fd, prompt) == -1) {
        perror("send");
        return "";
    }

    std::string response = receive_message(socket_fd);

    return response;
}
// !llama ---------------------------------------------------------------------------
int main() {
    int client_fd = start_client();

    if (client_fd == -1) {
        return 1;
    }

    std::string prompt;

    while (true) {
        std::cout << "You: ";
        if (!std::getline(std::cin, prompt)) {
            break;
        }

        if (prompt == "exit") {
            break;
        }

        if (prompt.empty()) {
            continue;
        }

        std::string response = send_prompt(client_fd, prompt);
        if (response.empty()) {
            std::cerr << "Server disconnected or returned an empty response\n";
            break;
        }

        std::cout << "Model: " << response << '\n';
    }

    close(client_fd);
    return 0;
}
