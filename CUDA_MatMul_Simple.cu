#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <locale.h>
#include <stdlib.h>

#define N 2000
#define div 200

__global__ void mulmat(int* a, int *b, int *c, int i) {
	int k = blockIdx.x * (N / div) + threadIdx.x;
	int j = blockIdx.y * (N / div) + threadIdx.y;
	int a_var = a[k * N + i];
	int b_var = b[i * N + j];
	c[k * N + j] += a_var * b_var;
}

int main()
{


	int* a, * b, * c;
	a = new int[N * N];
	b = new int[N * N];
	c = new int[N * N];
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			a[i * N + j] = rand() % 10;
			b[i * N + j] = rand() % 10;
			c[i * N + j] = 0;
		}
	}

	int* dev_a, * dev_b, * dev_c;
	cudaError_t cudaStatus;
	cudaMalloc((void**)&dev_a, N * N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * N * sizeof(int));

	cudaError_t error;

	error = cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	error = cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	error = cudaMemcpy(dev_c, c, N * N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}

	dim3 grid(div, div);
	dim3 blocks(N / div, N / div);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (int i = 0; i < N; ++i) {
		mulmat << <grid, blocks >> > (dev_a, dev_b, dev_c, i);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();

	error = cudaMemcpy(a, dev_a, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMemcpy(b, dev_b, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}

	error = cudaMemcpy(c, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	/*for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			printf("%d ", a[i * N + j]);
		}
		printf("\n");
	}
	printf("=======================================\n");
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			printf("%d ", b[i * N + j]);
		}
		printf("\n");
	}*/
	printf("%f milliseconds\n", milliseconds);

	/*for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%d ", a[i*N+j]);
		}
		printf("\n");
	}
	printf("---------------------------------------------------\n");
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%d ", b[i * N + j]);
		}
		printf("\n");
	}
	printf("---------------------------------------------------\n");
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%d ", c[i * N + j]);
		}
		printf("\n");
	}*/

	delete a;
	delete b;
	delete c;

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
