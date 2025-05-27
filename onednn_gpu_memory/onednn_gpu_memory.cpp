#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <dnnl.hpp>

size_t read_vram_usage_kb(const std::string& path = "/sys/class/drm/card0/mem_info_vram_used") {
    std::ifstream file(path);
    size_t used_bytes = 0;
    if (file.is_open()) {
        file >> used_bytes;
        file.close();
    } else {
        std::cerr << "Failed to open: " << path << std::endl;
    }
    return used_bytes / 1024; // KBに変換
}

void run_matmul() {
    using namespace dnnl;

    engine eng(engine::kind::gpu, 0);
    stream s(eng);

    const int M = 512, K = 512, N = 512;

    memory::dims a_dims = {M, K};
    memory::dims b_dims = {K, N};
    memory::dims c_dims = {M, N};

    auto a_md = memory::desc(a_dims, memory::data_type::f32, memory::format_tag::ab);
    auto b_md = memory::desc(b_dims, memory::data_type::f32, memory::format_tag::ab);
    auto c_md = memory::desc(c_dims, memory::data_type::f32, memory::format_tag::ab);

    auto a_mem = memory(a_md, eng);
    auto b_mem = memory(b_md, eng);
    auto c_mem = memory(c_md, eng);

    auto matmul_d = matmul::primitive_desc(eng, a_md, b_md, c_md);
    auto matmul_prim = matmul(matmul_d);

    std::unordered_map<int, memory> args = {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    };

    matmul_prim.execute(s, args);
    s.wait();
}

int main() {
    std::cout << "Initial GPU memory used: " << read_vram_usage_kb() << " KB" << std::endl;

    for (int i = 0; i < 10; ++i) {
        run_matmul();
        std::cout << "After iteration " << i + 1 << ": GPU memory used: " << read_vram_usage_kb() << " KB" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::cout << "Finished all matmul iterations." << std::endl;
    return 0;
}
