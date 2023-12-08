#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

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