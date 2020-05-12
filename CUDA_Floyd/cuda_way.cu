#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <locale.h>
#include <stdlib.h>

#define N 2000
#define inf 1000000
#define div 200

__global__ void floydCycle(int* b, int i) {
	int k = blockIdx.x*(N/div)+threadIdx.x;
	int j = blockIdx.y*(N/div)+threadIdx.y;
	int v1 = b[j * N + k];
	int v2 = b[j * N + i] + b[i * N + k];
	if (v1 > v2) {
		b[j * N + k] = v2;
	}
}

int main()
{
	// 3 матрицы A,B,C  C=A+B    NxN
	// каждая нить вычисляет 1 элемент из C - всего N^2


	int* a, * b;
	a = new int[N * N];
	b = new int[N * N];
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			if (i == j) {
				a[i * N + j] = 0;
			}
			else {
				if (rand() % 100 > 65) {
					a[i * N + j] = a[j * N + i] = b[i * N + j] = b[j * N + i] = rand() % 100;
				}
				else {
					a[i * N + j] = a[j * N + i] = -1;
					b[i * N + j] = b[j * N + i] = inf;
				}
			}
		}
	}

	int* dev_b;
	cudaError_t cudaStatus;
	cudaMalloc((void**)&dev_b, N * N * sizeof(int));

	cudaError_t error;


	error = cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}

	// Проба линейной архитектуры

	dim3 grid(div, div);
	dim3 blocks(N/div, N/div);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	// Внешний цикл алгоритма Флойду-Уоршела
	for (int i = 0; i < N; ++i) {
		floydCycle << <grid, blocks >> > (dev_b, i);
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

	error = cudaMemcpy(b, dev_b, N * N * sizeof(int), cudaMemcpyDeviceToHost);
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
	delete a;
	delete b;

	cudaFree(dev_b);
	return 0;
}