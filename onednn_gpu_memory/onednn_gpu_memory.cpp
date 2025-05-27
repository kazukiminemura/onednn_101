#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <level_zero/ze_api.h>
#include <iostream>
#include <vector>
#include <unistd.h> // sleep用

size_t get_level_zero_total_memory(const sycl::queue& queue) {
    auto dev = queue.get_device();
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    zeInit(0);
    uint32_t mem_count = 0;
    zeDeviceGetMemoryProperties(ze_dev, &mem_count, nullptr);
    std::vector<ze_device_memory_properties_t> mem_props(mem_count);
    zeDeviceGetMemoryProperties(ze_dev, &mem_count, mem_props.data());
    size_t total = 0;
    for (uint32_t i = 0; i < mem_count; ++i) {
        total += mem_props[i].totalSize;
    }
    return total;
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    size_t total_mem = get_level_zero_total_memory(queue);
    std::cout << "Level Zero GPU total memory: " << (total_mem / (1024 * 1024)) << " MB" << std::endl;

    dnnl::engine eng = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
    dnnl::stream s = dnnl::sycl_interop::make_stream(eng, queue);

    const int M = 512, K = 512, N = 512;
    const int LOOP = 10000; // ここを必要な回数に

    for (int iter = 0; iter < LOOP; ++iter) {
        // USMメモリ確保
        float* A = sycl::malloc_shared<float>(M * K, queue);
        float* B = sycl::malloc_shared<float>(K * N, queue);
        float* C = sycl::malloc_shared<float>(M * N, queue);
        std::fill(A, A + M*K, 1.0f);
        std::fill(B, B + K*N, 2.0f);
        std::fill(C, C + M*N, 0.0f);

        // メモリディスクリプタ
        auto a_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
        auto b_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
        auto c_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

        // oneDNNメモリ作成
        auto a_mem = dnnl::memory(a_md, eng, A);
        auto b_mem = dnnl::memory(b_md, eng, B);
        auto c_mem = dnnl::memory(c_md, eng, C);

        // matmul primitive作成
        auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md);
        auto matmul_prim = dnnl::matmul(matmul_pd);

        // 実行
        matmul_prim.execute(s, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem}
        });
        s.wait();

        // USMメモリ解放
        sycl::free(A, queue);
        sycl::free(B, queue);
        sycl::free(C, queue);

        // 進捗表示
        if (iter % 10 == 0) {
            std::cout << "Iteration " << iter << " done" << std::endl;
            // sleep(1); // 必要なら
        }
    }

    return 0;
}
