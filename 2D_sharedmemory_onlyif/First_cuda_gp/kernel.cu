
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define Section 12
#define Length 30//block块的长度方向
cudaError_t addWithCuda(float *T_Init, float dx, float dy, float tao, int nx, int ny, int tnpts, float *, float *, int num_blocks, int num_threadsx, int num_threadsy);
__device__ void Physicial_Parameters(float T, float *pho, float *Ce, float *lamd);
__device__ float Boundary_Condition(int j, int ny, float dx, float *ccml_zone, float *H_Init);

__global__ void addKernel(float *T_New, float *T_Last, float *ccml, float *H_Init, float dx, float dy, float tao, int nx, int ny, bool disout)
{
	//定义shared memory 
	const int M = Length, N = 11;//定义shared memory 大小
	__shared__ float shared[M+2][N];//只有一维是动态的，但必须在调用的时候声明大小。二维的就直接定义大小

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = threadIdx.y;
	int idx = j * ny + i;
	int D = ny;
	int tidx = threadIdx.x;
	int tid = threadIdx.x+1;//保证每个block边界温度计算无误
    //shared[tid][j]=0;//shared 赋初值
	float pho, Ce, lamd;
	float a, T_Up, T_Down, T_Right, T_Left, h = 200.0, Tw = 30.0, Vcast = -0.02, T_Cast = 1558.0;

	Physicial_Parameters(T_Last[idx], &pho, &Ce, &lamd);
	a = (lamd) / (pho*Ce);
	//h = Boundary_Condition(i, ny, dy, ccml, H_Init);

	//判断属于哪个区域
	int schedule;
	if (j == 0 && i != 0 && i != ny - 1) //1 
		schedule = 1;//第1条边
	if (j == nx - 1 && i != 0 && i != ny - 1)//2
		schedule = 2;//第2条边
	if (i == 0 && j != 0 && j != nx - 1)//3
		schedule = 3;//第3条边
	if (i == nx - 1 && j != 0 && j != nx - 1)//4
		schedule = 4;//第四条边
	if (i == 0 && j == 0)//5
		schedule = 3;//在3边上
	if (i == 0 && j == nx - 1)//6
		schedule = 3;//在3边上
	if (i == ny - 1 && j == 0)//7
		schedule = 4;//在4边上
	if (i == ny - 1 && j == nx - 1)//8
		schedule = 4;//在4边上
	if (i != 0 && i != ny - 1 && j != 0 && j != nx - 1)//9
		schedule = 0;//温度场内部

	//if (disout) {

		//赋值运算
		shared[tid][j] = T_Last[idx];//用shared 二维数组来表示二维温度场
		if (blockIdx.x > 0) {
			if (tidx == 0) shared[0][j] = T_Last[j*ny + blockDim.x*blockIdx.x - 1];
		}//保证所有block的左边界温度
		if (blockIdx.x < gridDim.x - 1) {
			if (tidx == 0)shared[M + 1][j] = T_Last[j*ny + blockDim.x*blockIdx.x + M + 1];
		}//保证所有block的右边界温度
		__syncthreads();	
		//对shared memory进行赋值之后要进行同步
		switch (schedule)
		{
		case 0:
			//T_New[idx] = T_Cast;
			T_Up = shared[tid][j + 1];
			T_Down = shared[tid][j - 1];
			T_Right = shared[tid + 1][j];
			T_Left = shared[tid - 1][j];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*shared[tid][j] +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
			//__syncthreads();
			break;
		case 1:
			//T_New[idx] = T_Cast;
			h = Boundary_Condition(i, ny, dy, ccml, H_Init);
			T_Up = shared[tid][j + 1];
			T_Down = shared[tid][j + 1] - 2 * dx * h * (shared[tid][j] - Tw) / lamd;
			T_Right = shared[tid + 1][j];
			T_Left = shared[tid - 1][j];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*shared[tid][j] +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
			//__syncthreads();
			break;
		case 2:
			//T_New[idx] = T_Cast;
			T_Up = shared[tid][j - 1];
			T_Down = shared[tid][j - 1];
			T_Right = shared[tid + 1][j];
			T_Left = shared[tid - 1][j];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*shared[tid][j] +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
			//__syncthreads();
			break;
		case 3:
			T_New[idx] = T_Cast;
			//__syncthreads();
			break;
		case 4:
			//T_New[idx] = T_Cast;
			T_Up = shared[tid][j + 1];
			T_Down = shared[tid][j - 1];
			T_Right = shared[tid - 1][j];
			T_Left = shared[tid - 1][j];//?
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*shared[tid][j] +
				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
			//__syncthreads();
			break;
		}
		//__syncthreads();
		T_Last[idx] = T_New[idx];
		//disout = !disout;
		//__syncthreads();
	//}
	//else
	//{
	//	Physicial_Parameters(T_New[idx], &pho, &Ce, &lamd);
	//	a = (lamd) / (pho*Ce);
	//	h = Boundary_Condition(i, ny, dy, ccml, H_Init);
	//	shared[tid][j] = T_New[idx];
	//	if (blockIdx.x  >0) {
	//		if(tidx==0)shared[0][j] = T_New[j*ny + blockDim.x*blockIdx.x - 1];
	//	}
	//	if (blockIdx.x < gridDim.x - 1) {
	//		if(tidx==0)shared[M + 1][j] = T_New[j*ny + blockDim.x*blockIdx.x + M + 1];
	//	}
	//	__syncthreads();
	//	switch (schedule)
	//	    {
	//		case 0:
	//			//T_Last[idx] = T_Cast;
	//			T_Up = shared[tid][j + 1];
	//			T_Down = shared[tid][j - 1];
	//			T_Right = shared[tid + 1][j];
	//			T_Left = shared[tid - 1][j];
	//			//T_New[idx] = T_Cast;
	//			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Last[idx] +
	//				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
	//			//__syncthreads();
	//			break;
	//		case 1:
	//			//T_Last[idx] = T_Cast;
	//			T_Up = shared[tid][j + 1];
	//			T_Down = shared[tid][j + 1] - 2 * dx * h * (shared[tid][j] - Tw) / lamd;
	//			T_Right = shared[tid + 1][j];
	//			T_Left = shared[tid - 1][j];
	//			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Last[idx] +
	//				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
	//			//__syncthreads();
	//			break;
	//		case 2:
	//			//T_Last[idx] = T_Cast;
	//			T_Up = shared[tid][j - 1];
	//			T_Down = shared[tid][j - 1];
	//			T_Right = shared[tid + 1][j];
	//			T_Left = shared[tid - 1][j];
	//			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Last[idx] +
	//				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
	//			//__syncthreads();
	//			break;
	//		case 3:
	//			T_Last[idx] = T_Cast;
	//			//__syncthreads();
	//			break;
	//		case 4:
	//			//T_Last[idx] = T_Cast;
	//			T_Up = shared[tid][j + 1];
	//			T_Down = shared[tid][j - 1];
	//			T_Right = shared[tid - 1][j];
	//			T_Left = shared[tid - 1][j];//?		
	//			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + (1 - 2 * a*(tao / (dx*dx)) - 2 * a*(tao / (dy*dy)) + tao*Vcast / dy)*T_Last[idx] +
	//				a*(tao / (dy*dy))*T_Right + (a*(tao / (dy*dy)) - tao*Vcast / dy)*T_Left;
	//			//__syncthreads();
	//			break;
	//		}
	//	//__syncthreads();
	//}
}

int main()
{
	const int nx = 11, ny = 3000, nz = 1;
	int num_blocks = 1, num_threadsx = 1, num_threadsy = 1, k = 0, tnpts = 10000.0;
	float T_Cast = 1558.0, Lx = 0.125, Ly = 28.599, dx, dy, t_final = 2000.0, tao;
	float *T_Init, *y;
	float ccml[Section + 1] = { 0.0,0.2,0.4,0.6,0.8,1.0925,2.27,4.29,5.831,9.6065,13.6090,19.87014,28.599 };
	float H_Init[Section] = { 1380,1170,980,800,1223.16,735.05,424.32,392.83,328.94,281.64,246.16,160.96 };

	T_Init = (float *)calloc(nx * ny, sizeof(float));

	num_threadsy = nx;
	num_threadsx = Length;
	num_blocks = ny / num_threadsx;

	for (int i = 0; i < nx; i++)
		for (int j = 0; j < ny; j++)
			T_Init[i * ny + j] = T_Cast;

	//printf("%d,\n ", k);
	dx = Lx / (nx - 1);
	dy = Ly / (ny - 1);
	tao = t_final / (tnpts - 1);

	printf("Casting Temperature = %f ", T_Cast);
	printf("\n");
	printf("The thick of steel billets(m) = %f ", Lx);
	printf("\n");
	printf("The length of steel billets(m) = %f ", Ly);
	printf("\n");
	printf("dx(m) = %f ", dx);
	printf("dy(m) = %f ", dy);
	printf("tao(s) = %f ", tao);
	printf("\n");
	printf("simulation time(s) = %f ", t_final);

	clock_t timestart = clock();
	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(T_Init, dx, dy, tao, nx, ny, tnpts, ccml, H_Init, num_blocks, num_threadsx, num_threadsy);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	clock_t timeend = clock();

	printf("running time = %d(mseconds)", (timeend - timestart));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

cudaError_t addWithCuda(float *T_Init, float dx, float dy, float tao, int nx, int ny, int tnpts, float *ccml, float *H_Init, int num_blocks, int num_threadsx, int num_threadsy)
{
	float *dev_T_New, *dev_T_Last, *dev_ccml, *dev_H_Init;
	float *T_Result;
	const int Num_Iter = 500;
	volatile bool dstOut = true;
	FILE *fp = NULL;

	T_Result = (float *)calloc(nx * ny, sizeof(float));

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_T_New, nx * ny * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_T_Last, nx * ny * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_ccml, (Section + 1) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_H_Init, Section * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_H_Init failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_T_Last, T_Init, nx * ny * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_T_Last failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_ccml, ccml, (Section + 1) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_ccml failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_H_Init, H_Init, Section * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_H_Init failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.

	dim3 threadsPerBlock(num_threadsx, num_threadsy);
	dim3 numBlocks(num_blocks);
	//int block_size = (num_threadsx + 2)*num_threadsy;
	// Launch a kernel on the GPU with one thread for each element.
	for (int i = 0; i < tnpts; i++)
    //for (int i = 0; i < 100; i++)
	{
		addKernel << <numBlocks, threadsPerBlock>> >(dev_T_New, dev_T_Last, dev_ccml, dev_H_Init, dx, dy, tao, nx, ny, dstOut);
		dstOut = !dstOut;

		if (i % Num_Iter == 0) {
		//if (i % 1 == 0) {
			cudaStatus = cudaMemcpy(T_Result, dev_T_Last, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy T_Result failed!");
				goto Error;
			}

			printf("time_step = %d \n", i);
			printf("%f, %f, %f", T_Result[0], T_Result[(nx - 1)*(ny - 1) - nx], T_Result[(nx - 1)*(ny - 1)]);	
			printf("\n");		
		}
		
	}

	fp = fopen("F:\\Temperature2DGPUStatic.txt", "w");
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		//fprintf(fp, " %f,%d", T_Result[i * ny + j], Ixy[i * ny + j]);
		{
			fprintf(fp, " %f", T_Result[i * ny + j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	//fp = fopen("F:\\TemperatureLat2DGPUStatic.txt", "w");
	//for (int i = 0; i < nx; i++)
	//{
	//	for (int j = 0; j < ny; j++)
	//		//fprintf(fp, " %f,%d", T_Result[i * ny + j], Ixy[i * ny + j]);
	//	{
	//		fprintf(fp, " %f", dev_T_Last[i * ny + j]);
	//	}
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);

	//fp = fopen("F:\\TemperatureNew2DGPUStatic.txt", "w");
	//for (int i = 0; i < nx; i++)
	//{
	//	for (int j = 0; j < ny; j++)
	//		//fprintf(fp, " %f,%d", T_Result[i * ny + j], Ixy[i * ny + j]);
	//	{
	//		fprintf(fp, " %f", dev_T_New[i * ny + j]);
	//	}
	//	fprintf(fp, "\n");
	//}
	//fclose(fp);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.


Error:
	cudaFree(dev_T_New);
	cudaFree(dev_T_Last);
	cudaFree(dev_ccml);
	cudaFree(dev_H_Init);

	return cudaStatus;
}
// Helper function for using CUDA to add vectors in parallel.

__device__ void Physicial_Parameters(float T, float *pho, float *Ce, float *lamd)
{
	float Ts = 1462.0, Tl = 1518.0, lamds = 30, lamdl = 50, phos = 7000, phol = 7500, ce = 540.0, L = 265600.0, fs = 0.0;
	if (T<Ts)
	{
		fs = 0;
		*pho = phos;
		*lamd = lamds;
		*Ce = ce;
	}

	if (T >= Ts&&T <= Tl)
	{
		fs = (T - Ts) / (Tl - Ts);
		*pho = fs*phos + (1 - fs)*phol;
		*lamd = fs*lamds + (1 - fs)*lamdl;
		*Ce = ce + L / (Tl - Ts);
	}

	if (T>Tl)
	{
		fs = 1;
		*pho = phol;
		*lamd = lamdl;
		*Ce = ce;
	}

}

__device__ float Boundary_Condition(int j, int ny, float dy, float *ccml_zone, float *H_Init)
{
	float YLabel, h = 0.0;
	YLabel = j*dy;

	for (int i = 0; i < Section; i++)
	{
		if (YLabel >= *(ccml_zone + i) && YLabel <= *(ccml_zone + i + 1))
			h = *(H_Init + i);
	}
	return h;
}