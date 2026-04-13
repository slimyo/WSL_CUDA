#pragma once

#include <cstdio>
#include <cstdint>  // CHAR_BIT
#include <fstream>
#include <string>   // std::to_string

#include <nv/target>

#include <thrust/fill.h>
#include <thrust/tabulate.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/universal_vector.h>  // 如果你实际用到它；否则可删

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/std/mdspan>

namespace ach {

static __host__ __device__ bool is_executed_on_gpu() {
  NV_IF_TARGET(NV_IS_HOST, (return false;));
  return true;
}

static __host__ __device__ const char* execution_space() {
  return is_executed_on_gpu() ? "GPU" : "CPU";
}

static double max_bandwidth() {
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);  // 保持原有调用

  // 改为用 cudaDeviceGetAttribute 查询（单位 kHz）
  int memory_clock_khz = 0;
  cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device);

  const std::size_t mem_freq = static_cast<std::size_t>(memory_clock_khz) * 1000;  // kHz → Hz

  const int bus_width = prop.memoryBusWidth;  // 这个字段还在，没被移除
  const std::size_t bytes_per_second = 2 * mem_freq * bus_width / CHAR_BIT;

  return static_cast<double>(bytes_per_second) / (1024.0 * 1024.0 * 1024.0);  // B/s → GB/s
}

__host__ __device__ void I_expect(const char* expected) {
  std::printf("expect %s; runs on %s;\n", expected, execution_space());
}

template <class ContainerT>
void store(int step, int height, int width, ContainerT& data) {
  std::ofstream file("/tmp/heat_" + std::to_string(step) + ".bin", std::ios::binary);
  file.write(reinterpret_cast<const char*>(&height), sizeof(int));
  file.write(reinterpret_cast<const char*>(&width), sizeof(int));
  file.write(reinterpret_cast<const char*>(thrust::raw_pointer_cast(data.data())),
             height * width * sizeof(float));
}

__host__ __device__ float compute(
    std::size_t cell_id,   // ← 改成 std::size_t 或 unsigned long
    cuda::std::mdspan<const float, cuda::std::dextents<std::size_t, 2>> temp) {  // ← index_type 改成 size_t
  auto height = temp.extent(0);
  auto width  = temp.extent(1);

  std::size_t column = cell_id % width;
  std::size_t row    = cell_id / width;

  if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
    float d2tdx2 = temp(row, column - 1) - 2 * temp(row, column) + temp(row, column + 1);
    float d2tdy2 = temp(row - 1, column) - 2 * temp(row, column) + temp(row + 1, column);
    return temp(row, column) + 0.2f * (d2tdx2 + d2tdy2);
  } else {
    return temp(row, column);
  }
}

thrust::device_vector<float> init(int height, int width) {
  thrust::device_vector<float> d_prev(height * width, 15.0f);
  thrust::fill_n(d_prev.begin(), width, 90.0f);
  thrust::fill_n(d_prev.begin() + width * (height - 1), width, 90.0f);
  return d_prev;
}

void simulate(int width, int height, const thrust::device_vector<float>& in,
              thrust::device_vector<float>& out, cudaStream_t stream) {
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);

  cub::DeviceTransform::Transform(
      thrust::make_counting_iterator(0),
      out.begin(),
      width * height,
      [=] __host__ __device__(int id) { return ach::compute(id, temp_in); },
      stream);
}

}  // namespace ach