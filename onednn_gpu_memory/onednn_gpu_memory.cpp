#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <level_zero/ze_api.h>
#include <iostream>
#include <vector>
#include <cassert>

size_t get_gpu_memory_total(ze_device_handle_t device) {
    uint32_t mem_count = 0;
    zeDeviceGetMemoryProperties(device, &mem_count, nullptr);
    std::vector<ze_device_memory_properties_t> mem_props(mem_count);
    zeDeviceGetMemoryProperties(device, &mem_count, mem_props.data());
    // ここでは最初のメモリバンクの総容量を返す
    return mem_props[0].totalSize;
}

int main() {
    zeInit(0);
    uint32_t driver_count = 0;
    zeDriverGet(&driver_count, nullptr);
    std::vector<ze_driver_handle_t> drivers(driver_count);
    zeDriverGet(&driver_count, drivers.data());

    uint32_t device_count = 0;
    zeDeviceGet(drivers[0], &device_count, nullptr);
    std::vector<ze_device_handle_t> devices(device_count);
    zeDeviceGet(drivers[0], &device_count, devices.data());
    ze_device_handle_t device = devices[0];

    sycl::queue queue(sycl::gpu_selector_v);
    dnnl::engine eng(dnnl::engine::kind::gpu, 0);
    dnnl::stream s(eng);

    const int M = 512, K = 512, N = 512;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    auto a_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    auto b_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    auto c_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

    auto a_mem = dnnl::memory(a_md, eng, A.data());
    auto b_mem = dnnl::memory(b_md, eng, B.data());
    auto c_mem = dnnl::memory(c_md, eng, C.data());

    // 修正版: matmul primitive_descの作成
    auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md);
    auto matmul_prim = dnnl::matmul(matmul_pd);

    size_t mem_total = get_gpu_memory_total(device);
    std::cout << "GPU total memory: " << mem_total / (1024 * 1024) << " MB" << std::endl;

    matmul_prim.execute(s, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    });
    s.wait();

    return 0;
}
