#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#define QX_DEF_CHAR_MAX 255

// void invokePrelimHost(
//     unsigned char *img, int width, int height, int channel,
//     float sigma_spatial, float sigma_range)
// {
//   // REFACTOR: store alpha_ of each pixel in map_factor_b for l->r, map_factor_c for r->l, use wh threads

//   float *map_factor_b = new float[width * height];
//   float *map_factor_c = new float[width * height];
//   // range table
//   float range_table[QX_DEF_CHAR_MAX + 1];
//   float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
//   for (int i = 0; i <= QX_DEF_CHAR_MAX; i++)
//     range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));
//   float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * width)));

//   for (int y = 0; y < height; y++)
//   {
//     unsigned char *texture_x = &img[y * width * channel];
//     float *map_alphas_l = &map_factor_b[y * width];
//     float *map_alphas_r = &map_factor_c[y * width];
//     for (int x = 1; x < width; x++)
//     {
//       unsigned char tpr = texture_x[x * 3 - 3], tcr = texture_x[x * 3];
//       unsigned char tpg = texture_x[x * 3 - 2], tcg = texture_x[x * 3 + 1];
//       unsigned char tpb = texture_x[x * 3 - 1], tcb = texture_x[x * 3 + 2];
//       unsigned char dr = abs(tcr - tpr);
//       unsigned char dg = abs(tcg - tpg);
//       unsigned char db = abs(tcb - tpb);
//       int range_dist = (((dr << 1) + dg + db) >> 2);
//       float weight = range_table[range_dist];
//       float alpha_ = weight * alpha;
//       map_alphas_l[x] = alpha_;
//       range_dist = (((db << 1) + dg + dr) >> 2);
//       map_alphas_r[x] = range_table[range_dist] * alpha;
//     }
//   }

//   // Print the values of map_factor_b_h
//   printf("Values of map_factor_b:\n");
//   for (int i = 0; i < height; i++)
//   {
//     for (int j = 0; j < width; j++)
//     {
//       printf("%.8f ", map_factor_b[i * width + j]);
//     }
//     printf("\n");
//   }

//   // Print the values of map_factor_c_h
//   printf("\nValues of map_factor_c:\n");
//   for (int i = 0; i < height; i++)
//   {
//     for (int j = 0; j < width; j++)
//     {
//       printf("%.8f ", map_factor_c[i * width + j]);
//     }
//     printf("\n");
//   }
// }

__global__ void prelimKernel(
    unsigned char *img, float *map_factor_b, float *map_factor_c,
    float *range_table, int width, int height, int channel,
    float sigma_spatial);

void invokePrelimKernel(
    unsigned char *img_h, int width, int height, int channel,
    float sigma_spatial, float sigma_range, int rows_per_block)
{
  // device allocation
  // map_factor_b
  float *map_factor_b_d;
  cudaMalloc((void **)&map_factor_b_d, width * height * sizeof(float));
  if (!map_factor_b_d)
  {
    printf("Prelim Kernel: Cuda malloc fail on map_factor_b_d");
    exit(1);
  }

  // map_factor_c
  float *map_factor_c_d;
  cudaMalloc((void **)&map_factor_c_d, width * height * sizeof(float));
  if (!map_factor_c_d)
  {
    printf("Prelim Kernel: Cuda malloc fail on map_factor_c_d");
    cudaFree(map_factor_b_d);
    exit(1);
  }

  // img_d
  unsigned char *img_d;
  cudaMalloc((void **)&img_d, height * width * channel * sizeof(char));
  if (!img_d)
  {
    printf("Prelim Kernel: Cuda malloc fail on img_d");
    cudaFree(map_factor_b_d);
    cudaFree(map_factor_c_d);
    exit(1);
  }

  // range table
  float *range_table_d;
  cudaMalloc((void **)&range_table_d, (QX_DEF_CHAR_MAX + 1) * sizeof(float));
  if (!range_table_d)
  {
    printf("Prelim Kernel: Cuda malloc fail on range_table_d");
    cudaFree(map_factor_b_d);
    cudaFree(map_factor_c_d);
    cudaFree(img_d);
    exit(1);
  }

  // host allocation
  // map_factor_b
  float *map_factor_b_h = new float[width * height];

  // map_factor_c
  float *map_factor_c_h = new float[width * height];

  // range table
  float range_table[QX_DEF_CHAR_MAX + 1];
  float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
  for (int i = 0; i <= QX_DEF_CHAR_MAX; i++)
    range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));

  cudaMemcpy(img_d, img_h, height * width * channel * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(range_table_d, range_table, (QX_DEF_CHAR_MAX + 1) * sizeof(float), cudaMemcpyHostToDevice);

  // kernel params
  int total_threads = height;
  int threads_per_block = rows_per_block;
  int num_blocks;
  if (total_threads % threads_per_block == 0)
    num_blocks = total_threads / threads_per_block;
  else
    num_blocks = total_threads / threads_per_block + 1;

  dim3 grid(num_blocks, 1, 1);
  dim3 block(threads_per_block, 1, 1);

  prelimKernel<<<num_blocks, threads_per_block>>>(
      img_d, map_factor_b_d, map_factor_c_d, range_table_d,
      width, height, channel, sigma_spatial);

  cudaDeviceSynchronize();
  cudaMemcpy(map_factor_b_h, map_factor_b_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(map_factor_c_h, map_factor_c_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the values of map_factor_b_h
  printf("Values of map_factor_b_h:\n");
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      printf("%.8f ", map_factor_b_h[i * width + j]);
    }
    printf("\n");
  }

  // Print the values of map_factor_c_h
  printf("\nValues of map_factor_c_h:\n");
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      printf("%.8f ", map_factor_c_h[i * width + j]);
    }
    printf("\n");
  }
}

__global__ void prelimKernel(
    unsigned char *img, float *map_factor_b, float *map_factor_c,
    float *range_table, int width, int height, int channel,
    float sigma_spatial)
{
  int row_number = blockIdx.x * blockDim.x + threadIdx.x;

  if (row_number < height)
  {
    unsigned char *texture_x = &img[row_number * width * channel];
    float *map_alphas_l = &map_factor_b[row_number * width];
    float *map_alphas_r = &map_factor_c[row_number * width];

    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * width)));
    for (int x = 1; x < width; x++)
    {
      unsigned char tpr = texture_x[row_number * width * channel + (x - 1) * 3];
      unsigned char tcr = texture_x[row_number * width * channel + x * 3];
      unsigned char tpg = texture_x[row_number * width * channel + (x - 1) * 3 + 1];
      unsigned char tcg = texture_x[row_number * width * channel + x * 3 + 1];
      unsigned char tpb = texture_x[row_number * width * channel + (x - 1) * 3 + 2];
      unsigned char tcb = texture_x[row_number * width * channel + x * 3 + 2];

      unsigned char dr = abs(tcr - tpr);
      unsigned char dg = abs(tcg - tpg);
      unsigned char db = abs(tcb - tpb);

      int range_dist = (((dr << 1) + dg + db) >> 2);
      float weight = range_table[range_dist];
      float alpha_ = weight * alpha;

      map_alphas_l[x] = alpha_;

      range_dist = (((db << 1) + dg + dr) >> 2);
      map_alphas_r[x] = range_table[range_dist] * alpha;
    }
  }
}

int main()
{
  // Input image data
  unsigned char img_h[] = {
      10, 20, 30,    // Row 1, Pixel 1, Grayscale value: 10
      40, 50, 60,    // Row 1, Pixel 2, Grayscale value: 40
      70, 80, 90,    // Row 1, Pixel 3, Grayscale value: 70
      100, 110, 120, // Row 2, Pixel 1, Grayscale value: 100
      130, 140, 150, // Row 2, Pixel 2, Grayscale value: 130
      160, 170, 180, // Row 2, Pixel 3, Grayscale value: 160
  };

  int width = 3;
  int height = 2;
  int channels = 1; // Grayscale image

  float sigma_spatial = 100;
  float sigma_range = 100;
  int rows_per_block = 2;

  invokePrelimKernel(img_h, width, height, channels, sigma_spatial, sigma_range, rows_per_block);
  // invokePrelimHost(img_h, width, height, channels, sigma_spatial, sigma_range);

  return 0;
}