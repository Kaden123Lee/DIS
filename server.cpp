#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <cstdint>
#include <limits>
#include <vector>
#include <string>
#include "llama.h"
#include "socket_messages.h"
// !llama ---------------------------------------------------------------------------
static llama_model* model = nullptr;
static llama_context* ctx = nullptr;

bool initialize_model(const std::string& model_path) {
    // Initialize llama.cpp
    llama_backend_init();

    // Load the model
    llama_model_params model_params = llama_model_default_params();

    model = llama_model_load_from_file(
        model_path.c_str(),
        model_params
    );

    if (model == nullptr) {
        std::cerr << "Failed to load model\n";
        llama_backend_free();
        return false;
    }

    // Create the inference context
    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 512;

    ctx = llama_init_from_model(model, ctx_params);

    if (ctx == nullptr) {
        std::cerr << "Failed to create context\n";
        llama_model_free(model);
        model = nullptr;
        llama_backend_free();
        return false;
    }

    return true;
}
// ! ---------------------------------------------------------------------------------------------
std::string run_inference(const std::string& prompt) {
    if (model == nullptr || ctx == nullptr) {
        return "Error: Model not initialized";
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // 1. Start a fresh request
    llama_memory_clear(llama_get_memory(ctx), true);

    // 2. Format the prompt for our chat model
    const char* chat_template = llama_model_chat_template(model, nullptr);

    llama_chat_message message = {"user", prompt.c_str()};

    int formatted_size = llama_chat_apply_template(
        chat_template, &message, 1, true, nullptr, 0
    );

    if (formatted_size < 0) {
        return "Error: Chat formatting failed";
    }

    std::vector<char> formatted(formatted_size + 1);

    if (llama_chat_apply_template(
            chat_template, &message, 1, true,
            formatted.data(), formatted.size()) < 0) {
        return "Error: Chat formatting failed";
    }

    // 3. Convert text into tokens
    int token_count = -llama_tokenize(
        vocab, formatted.data(), formatted_size,
        nullptr, 0, true, true
    );

    if (token_count <= 0 || token_count > 512) {
        return "Error: Prompt too long or invalid";
    }

    std::vector<llama_token> tokens(token_count);

    if (llama_tokenize(
            vocab, formatted.data(), formatted_size,
            tokens.data(), tokens.size(), true, true) < 0) {
        return "Error: Tokenization failed";
    }

    // 4. Create a batch and sampler
    llama_batch_ext* batch = llama_batch_ext_init(ctx);

    llama_sampler* sampler =
        llama_sampler_chain_init(llama_sampler_chain_default_params());

    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // 5. Process the prompt
    llama_batch_ext_clear(batch);

    for (int i = 0; i < token_count; i++) {
        int index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        llama_pos position = i;

        llama_batch_ext_set_pos(batch, index, &position);
    }

    llama_batch_ext_set_output_logits(batch, token_count - 1, true);

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
        llama_sampler_free(sampler);
        llama_batch_ext_free(batch);
        return "Error: Prompt processing failed";
    }

    // 6. Generate response tokens
    std::string response;
    int position = token_count;

    for (int i = 0; i < 512 && position < llama_n_ctx(ctx); i++) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        // Convert generated token into text
        char buffer[256];

        int length = llama_token_to_piece(
            vocab, token, buffer, sizeof(buffer), 0, true
        );

        if (length < 0) {
            break;
        }

        response.append(buffer, length);

        // Process this token to generate the next one
        llama_batch_ext_clear(batch);

        int index = llama_batch_ext_add_token(batch, 0, token);
        llama_batch_ext_set_pos(batch, index, &position);
        llama_batch_ext_set_output_logits(batch, index, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
            break;
        }

        position++;
    }

    // 7. Release temporary resources
    llama_sampler_free(sampler);
    llama_batch_ext_free(batch);

    return response;
}
// ! -------------------------------------------------------------
void cleanup_model() {
    if (ctx != nullptr) {
        llama_free(ctx);
        ctx = nullptr;
    }

    if (model != nullptr) {
        llama_model_free(model);
        model = nullptr;
    }

    llama_backend_free();
}

/*
| Function | Purpose | When called |
|---|---|---|
| `initialize_model()` | Load the 7B model and initialize llama.cpp | Once at startup |
| `run_inference()` | Generate a response from a prompt | Every request |
| `cleanup_model()` | Free model memory and resources | At shutdown |
*/

// !llama ---------------------------------------------------------------------------
int start_server() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        perror("socket");
        return -1;
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(8080);
    address.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == -1) {
        perror("bind");
        close(server_fd);
        return -1;
    }
    if (listen(server_fd, 5) == -1) {
        perror("listen");
        close(server_fd);
        return -1;
    }

    return server_fd;
}

int handle_client(int server_fd) {
    std::cout << "Waiting for client...\n";

    int client_fd = accept(server_fd, nullptr, nullptr);

    if (client_fd == -1) {
        perror("accept");
        return -1;
    }

    while (true) {
        // 1. Receive prompt from client
        std::string prompt = receive_message(client_fd);

        // 2. Stop if client disconnected
        if (prompt.empty()) {
            break;
        }

        // 3. Generate response using llama.cpp
        std::string response = run_inference(prompt);

        // 4. Send response back to client
        if (send_message(client_fd, response) == -1) {
            break;
        }
    }

    close(client_fd);
    return 0;
}

int main() {
    // 1. Load the 7B model
    if (!initialize_model(
        "llama.cpp/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    )) {
        return 1;
    }

    // 2. Start TCP server
    int server_fd = start_server();

    if (server_fd == -1) {
        cleanup_model();
        return 1;
    }

    // 3. Handle client requests
    int result = handle_client(server_fd);

    // 4. Clean up
    close(server_fd);
    cleanup_model();

    return result == -1 ? 1 : 0;
}
