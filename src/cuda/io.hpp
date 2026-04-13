#pragma once
#include <fstream>
#include <string>
#include <thrust/host_vector.h>

namespace ach {

inline void store(int step, int height, int width,
                  const thrust::host_vector<float>& data) {
  std::ofstream file("/tmp/heat_" + std::to_string(step) + ".bin",
                     std::ios::binary);
  file.write(reinterpret_cast<const char*>(&height), sizeof(int));
  file.write(reinterpret_cast<const char*>(&width), sizeof(int));
  file.write(reinterpret_cast<const char*>(data.data()),
             height * width * sizeof(float));
}

}