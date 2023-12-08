#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <cuda.h>

#include "updateImageRow.cu"

// CPU version of the code
void cpuUpdateImageRow(float *img_out_f, int width, int height, int channel, float *img_temp, float *slice_factor_b, float *map_factor_b)
{
  int k = 0;
  int h1 = height - 1;
  float *ypy = slice_factor_b;
  memcpy(ypy, &img_temp[h1 * width * channel], sizeof(float) * width * channel);

  for (int x = 0; x < width; x++)
  {
    for (int c = 0; c < channel; c++)
    {
      int idx = (h1 * width + x) * channel + c;
      img_out_f[idx] = 0.5f * (img_out_f[idx] + ypy[k++]) / map_factor_b[h1 * width + x];
    }
  }
}

// CUDA kernel version
__global__ void updateImageRow(float *buffer, int width, int height, int channel)
{
  float *img_out_f = buffer;
  int h1 = height - 1;
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId >= width * channel)
    return;
  float *img_temp = &img_out_f[width * height * channel];
  float *map_factor_a = &img_temp[width * height * channel];
  float *map_factor_b = &map_factor_a[width * height];
  int x = threadId % width;
  int c = threadId % channel;
  int idx = (h1 * width + x) * channel + c;
  float *ypy = &img_temp[h1 * width * channel];
  img_out_f[idx] = 0.5f * (img_out_f[idx] + ypy[threadId]) / map_factor_b[h1 * width + x];
};

// Function to compare CPU and CUDA results
bool compareResults(float *cpuResult, float *cudaResult, int size)
{
  const float epsilon = 1e-5;

  for (int i = 0; i < size; i++)
  {
    if (std::fabs(cpuResult[i] - cudaResult[i]) > epsilon)
    {
      return false;
    }
  }
  return true;
}

int main()
{
  const int width = 4;
  const int height = 3;
  const int channel = 1;

  int buffer_size = (width * height * channel + width * height + width * channel) * 2;
  float *buffer = new float[buffer_size];
  // Assign values to the buffer array
  for (int i = 0; i < buffer_size; i++)
  {
    buffer[i] = i * 1.0f;
  }

  float *img_out_f = buffer;
  float *img_temp = &img_out_f[width * height * channel];
  float *map_factor_a = &img_temp[width * height * channel];
  float *map_factor_b = &map_factor_a[width * height];
  float *slice_factor_a = &map_factor_b[width * height];
  float *slice_factor_b = &slice_factor_a[width * channel];

  // Allocate GPU memory
  float *buffer_d;
  cudaMalloc(&buffer_d, buffer_size * sizeof(float));
  // Copy data to GPU
  cudaMemcpy(buffer_d, buffer, buffer_size * sizeof(float), cudaMemcpyHostToDevice);

  // Run CPU version
  cpuUpdateImageRow(img_out_f, width, height, channel, img_temp, slice_factor_b, map_factor_b);

  // Run CUDA version
  // Specify block and grid dimensions
  int total_threads = width * channel;
  int blockSize = 128;
  int num_blocks;
  if (total_threads % blockSize == 0)
    num_blocks = total_threads / blockSize;
  else
    num_blocks = total_threads / blockSize + 1;

  updateImageRow<<<num_blocks, blockSize>>>(buffer_d, width, height, channel);
  cudaDeviceSynchronize();

  // Copy data back to CPU
  float *buffer_gpu = new float[buffer_size];
  cudaMemcpy(buffer_gpu, buffer_d, buffer_size * sizeof(float), cudaMemcpyDeviceToHost);

  // Compare results
  float *img_out_f_gpu = buffer_gpu;
  int size = width * height * channel;
  bool resultMatch = compareResults(img_out_f, img_out_f_gpu, size);

  if (resultMatch)
  {
    std::cout << "Results match!\n";
  }
  else
  {
    std::cout << "Results do not match!\n";
  }

  // Print img_out_f
  std::cout << "img_out_f:\n";
  for (int i = 0; i < size; i++)
  {
    std::cout << img_out_f[i] << " ";
  }
  std::cout << "\n";

  // Print img_out_f_gpu
  std::cout << "img_out_f_gpu:\n";
  for (int i = 0; i < size; i++)
  {
    std::cout << img_out_f_gpu[i] << " ";
  }
  std::cout << "\n";

  // Clean up
  delete[] buffer;

  cudaFree(buffer_d);

  return 0;
}
