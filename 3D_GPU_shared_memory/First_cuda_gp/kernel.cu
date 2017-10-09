//3D_shared_memory
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "book.h"
#include "gridcheck.h"

# define Section 12  // number of cooling sections
# define num_threadsx 3
# define num_threadsy 15
# define num_threadsz 21
# define num_blocksx 7
# define num_blocksy 200
# define num_blocksz 1
cudaError_t addWithCuda(float *T_Init, float dx, float dy, float dz, float tao, int nx, int ny, int nz, int tnpts, float *, float *);
__device__ void Physicial_Parameters(float T, float *pho, float *Ce, float *lamd);
__device__ float Boundary_Condition(int j, float dx, float *ccml_zone, float *H_Init);

__global__ void addKernel(float *T_New, float *T_Last, float *ccml, float *H_Init, float dx, float dy, float dz, float tao, int nx, int ny, int nz, bool disout)
{
	//定义shared memory 
	const int M = num_threadsx, N = num_threadsy, Z = num_threadsz;//定义shared memory 大小
	__shared__ float shared[M+2][N+2][Z+2];//只有一维是动态的，但必须在调用的时候声明大小。多维的就直接定义大小

	int tidx = threadIdx.x;
	int posx = threadIdx.x + 1;
	int tidy = threadIdx.y;
	int posy = threadIdx.y+1;
	int tidz = threadIdx.z;
	int posz = threadIdx.z + 1;
	int i = blockIdx.x*blockDim.x+tidx;
	int j = blockIdx.y*blockDim.y + tidy;
	int m = blockIdx.z*blockDim.z + tidz;
	int idx = j * nx * nz + m * nx + i;
	int ND = nx * nz;
	int D = nx;

	float pho, Ce, lamd; // physical parameters pho represents desity, Ce is specific heat and lamd is thermal conductivity
	float a, T_Up, T_Down, T_Right, T_Left, T_Forw, T_Back, h = 100.0, Tw = 30.0, Vcast = -0.02, T_Cast = 1558.0;


	//判断区域
	int schedule;
	if (j == 0) //1面
		schedule = 1;
	if (j == (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //10面
		schedule = 2;
	if (j == (ny - 1) && i == 0 && m != 0 && m != (nz - 1)) //11边
		//schedule = 2;
		schedule = 3;
    if (j == (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //12边
		//schedule = 2;
		schedule = 3;
	if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //13边
		schedule = 3;
		//schedule = 2;
	if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //14边
		//schedule = 2;
		schedule = 3;
	if (j == (ny - 1) && i == 0 && m == 0)  //15点
		//schedule = 2;
		schedule = 3;
	if (j == (ny - 1) && i == 0 && m == (nz - 1))  //16点
		//schedule = 2;
		schedule = 3;
	if (j == (ny - 1) && i == (nx - 1) && m == 0)  //17点
		//schedule = 2;
		schedule = 3;
	if (j == (ny - 1) && i == (nx - 1) && m == (nz - 1))  //18点
		//schedule = 2;
		schedule = 3;
	if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //19面
		schedule = 3;
	if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //20面
		schedule = 4;
	if (j != 0 && j != (ny - 1) && i == 0 && m == 0) //21边
		schedule = 3;
	if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == 0)  //22边
		schedule = 3;
	if (j != 0 && j != (ny - 1) && i == 0 && m == (nz - 1)) //23边
		//schedule = 4;
		schedule = 3;
	if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == (nz - 1)) //24边
		//schedule = 4;
    	schedule = 3;
	if (j != 0 && j != (ny - 1) && i == 0 && m != 0 && m != (nz - 1))  //25边
		schedule = 5;
	if (j != 0 && j != (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //26面
		schedule = 6;
	if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1))//27内部
		schedule = 0;
	//赋值语句
	shared[posx][posy][posz] = T_Last[idx];//用shared 三维数组来表示三维温度场

	if (blockIdx.x > 0) {
		if(threadIdx.x==0) shared[0][posy][posz] = T_Last[j*ND+m*D+i-1];
	}//保证所有blockx的左边界温度
	if (blockIdx.x < gridDim.x-1) {
		if (threadIdx.x == 0) shared[M + 1][posy][posz] = T_Last[j*ND + m*D +i+ M + 1];
	}//保证所有blockx的右边界温度
	if (blockIdx.y > 0) {
		if (threadIdx.y == 0) shared[posx][0][posz] = T_Last[(j-1)*ND + m*D +i];
	}//保证所有blockY的左边界温度
	if (blockIdx.x < gridDim.y - 1) {
		if (threadIdx.y == 0) shared[posx][N+1][posz] = T_Last[(j+N+1)*ND + m*D + i];
	}//保证所有blockY的右边界温度
	if (blockIdx.z > 0) {
		if (threadIdx.z == 0) shared[posx][posy][0] = T_Last[j*ND + (m-1)*D +i];
	}//保证所有blockx的左边界温度
	if (blockIdx.x < gridDim.z - 1) {
		if (threadIdx.z == 0) shared[posx][posy][Z+1] = T_Last[j*ND + (m+Z+1)*D  + i];
	}//保证所有blockx的右边界温度
	__syncthreads();

	Physicial_Parameters(T_New[idx], &pho, &Ce, &lamd);
	a = (lamd) / (pho*Ce);
	//h = Boundary_Condition(j, dy, ccml, H_Init);
	switch (schedule)
	{
	case 0:
		//T_New[idx] = T_Cast;
		T_Up = shared[posx+1][posy][posz];// T_Last[idx + 1];
		T_Down = shared[posx-1][posy][posz];// T_Last[idx - 1];
		T_Right = shared[posx][posy+1][posz];// T_Last[idx + ND];
		T_Left = shared[posx][posy-1][posz]; //T_Last[idx - ND];
		T_Forw = shared[posx][posy][posz+1]; //T_Last[idx + D];
		T_Back = shared[posx][posy][posz-1]; //T_Last[idx - D];
		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*shared[posx][posy][posz]
			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
			break;
	case 1:
		T_New[idx] = T_Cast;
		break;
	case 2:
		//T_New[idx] = T_Cast;

		T_Up = shared[posx+1][posy][posz];// T_Last[idx + 1];
		T_Down = shared[posx-1][posy][posz];//  T_Last[idx - 1];
		T_Right = shared[posx][posy-1][posz];//  T_Last[idx - ND];
		T_Left = shared[posx][posy-1][posz];// T_Last[idx - ND];
		T_Forw = shared[posx][posy][posz+1];// T_Last[idx + D];
		T_Back = shared[posx][posy][posz-1];// T_Last[idx - D];
		T_New[idx] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*shared[posx][posy][posz]
			+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
			break;
	case 3:
		//T_New[idx] = T_Cast;		
		h = Boundary_Condition(j, dy, ccml, H_Init);
		T_Up = shared[posx+1][posy][posz];//T_Last[idx + 1];
		T_Down = shared[posx-1][posy][posz];//T_Last[idx - 1];
		T_Right = shared[posx][posy+1][posz];// T_Last[idx + ND];
		T_Left = shared[posx][posy-1][posz];//T_Last[idx - ND];
		T_Forw = shared[posx][posy][posz+1];//T_Last[idx + D];
		T_Back = shared[posx][posy][posz+1] - 2 * dz * h * (shared[posx][posy][posz] - Tw) / lamd;// T_Last[idx + D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*shared[posx][posy][posz]
			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		break;
	case 4:
		//T_New[idx] = T_Cast;		
		T_Up = shared[posx+1][posy][posz];//T_Last[idx + 1];
		T_Down = shared[posx+1][posy][posz];// T_Last[idx + 1];
		T_Right = shared[posx][posy+1][posz];//T_Last[idx + ND];
		T_Left = shared[posx][posy-1][posz];//T_Last[idx - ND];
		T_Forw = shared[posx][posy][posz-1];//T_Last[idx - D];
		T_Back = shared[posx][posy][posz-1];//T_Last[idx - D];
		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*shared[posx][posy][posz]
			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		break;
	case 5:
		//T_New[idx] = T_Cast;
		h = Boundary_Condition(j, dy, ccml, H_Init);
		T_Up = shared[posx+1][posy][posz];// T_Last[idx + 1];
		T_Down = shared[posx+1][posy][posz] - 2 * dx * h * (shared[posx][posy][posz] - Tw) / lamd;//T_Last[idx + 1] - 2 * dx * h * (T_Last[idx] - Tw) / lamd;
		T_Right = shared[posx][posy+1][posz];// T_Last[idx + ND];
		T_Left = shared[posx][posy-1][posz];// T_Last[idx - ND];
		T_Forw = shared[posx][posy][posz+1];//T_Last[idx + D];
		T_Back = shared[posx][posy][posz-1];// T_Last[idx - D];
		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*shared[posx][posy][posz]
			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		break;
	case 6:
		//T_New[idx] = T_Cast;
		h = Boundary_Condition(j, dy, ccml, H_Init);
		T_Up = shared[posx-1][posy][posz] - 2 * dx * h * (shared[posx][posy][posz] - Tw) / lamd;// T_Last[idx - 1] - 2 * dx * h * (T_Last[idx] - Tw) / lamd;
		T_Down = shared[posx-1][posy][posz];// T_Last[idx - 1];
		T_Right = shared[posx][posy+1][posz];// T_Last[idx + ND];
		T_Left = shared[posx][posy-1][posz];// T_Last[idx - ND];
		T_Forw = shared[posx][posy][posz+1];// T_Last[idx + D];
		T_Back = shared[posx][posy][posz-1];// T_Last[idx - D];
		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*shared[posx][posy][posz]
			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		break;
	}
	T_Last[idx] = T_New[idx];
	//if (disout) {
	//	Physicial_Parameters(T_New[idx], &pho, &Ce, &lamd);
	//	a = (lamd) / (pho*Ce);
	//	h = Boundary_Condition(j, dy, ccml, H_Init);
	//	if (j == 0) //1
	//	{
	//		T_New[idx] = T_Cast;
	//	}

	//	else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //10
	//	{
	//		//T_New[idx] = 1550.0;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx - ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == 0 && m != 0 && m != (nz - 1)) //11
	//	{
	//		//T_New[idx] = 1550.0;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx + 1];
	//		T_Right = T_Last[idx - ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //12
	//	{
	//		//T_New[idx] = 1550.0;
	//		T_Up = T_Last[idx - 1];
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx - ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //13
	//	{
	//		//T_New[idx] = 1550.0;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx - ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx + D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //14
	//	{
	//		//T_New[idx] = 1550.0;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx - ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx - D];
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == 0 && m == 0)  //15
	//	{
	//		//T_New[idx] = 1550.0;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx + 1];
	//		T_Right = T_Last[idx - ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx + D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == 0 && m == (nz - 1))  //16
	//	{
	//		//T_New[idx] = 1550.0;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx + 1];
	//		T_Right = T_Last[idx - ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx - D];
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == (nx - 1) && m == 0)  //17
	//	{
	//		//T_New[idx] = 1550.0;
	//		T_Up = T_Last[idx - 1];
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx - ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx + D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == (nx - 1) && m == (nz - 1))  //18
	//	{
	//		//T_New[idx] = 1550.0;
	//		T_Up = T_Last[idx - 1];
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx - ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx - D];
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //19
	//	{
	//		//T_New[idx] = T_Cast;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx + ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx + D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //20
	//	{
	//		//T_New[idx] = T_Cast;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx + ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx - D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == 0 && m == 0) //21
	//	{
	//		//T_New[idx] = T_Cast;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx + 1];
	//		T_Right = T_Last[idx + ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx + D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == 0)  //22
	//	{
	//		//T_New[idx] = T_Cast;
	//		T_Up = T_Last[idx - 1];
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx + ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx + D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == 0 && m == (nz - 1)) //23
	//	{
	//		//T_New[idx] = T_Cast;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx + 1];
	//		T_Right = T_Last[idx + ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx - D];
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == (nz - 1)) //24
	//	{
	//		//T_New[idx] = T_Cast;
	//		T_Up = T_Last[idx - 1];
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx + ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx - D];
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == 0 && m != 0 && m != (nz - 1))  //25
	//	{
	//		//T_New[idx] = T_Cast;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx + 1] - 2 * dx * h * (T_Last[idx] - Tw) / lamd;
	//		T_Right = T_Last[idx + ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //26
	//	{
	//		//T_New[idx] = T_Cast;
	//		T_Up = T_Last[idx - 1] - 2 * dx * h * (T_Last[idx] - Tw) / lamd;
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx + ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else  //27
	//	{
	//		//T_New[idx] = T_Cast;
	//		T_Up = T_Last[idx + 1];
	//		T_Down = T_Last[idx - 1];
	//		T_Right = T_Last[idx + ND];
	//		T_Left = T_Last[idx - ND];
	//		T_Forw = T_Last[idx + D];
	//		T_Back = T_Last[idx - D];
	//		T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}
	//}

	//else
	//{
	//	Physicial_Parameters(T_New[idx], &pho, &Ce, &lamd);
	//	a = (lamd) / (pho*Ce);
	//	h = Boundary_Condition(j, dy, ccml, H_Init);
	//	if (j == 0) //1
	//	{
	//		T_Last[idx] = T_Cast;
	//	}

	//	else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //10
	//	{
	//		//T_Last[idx] = 1550.0;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx - ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == 0 && m != 0 && m != (nz - 1)) //11
	//	{
	//		//T_Last[idx] = 1550.0;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx + 1];
	//		T_Right = T_New[idx - ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //12
	//	{
	//		//T_Last[idx] = 1550.0;
	//		T_Up = T_New[idx - 1];
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx - ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //13
	//	{
	//		//T_Last[idx] = 1550.0;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx - ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx + D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //14
	//	{
	//		//T_Last[idx] = 1550.0;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx - ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx - D];
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == 0 && m == 0)  //15
	//	{
	//		//T_Last[idx] = 1550.0;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx + 1];
	//		T_Right = T_New[idx - ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx + D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == 0 && m == (nz - 1))  //16
	//	{
	//		//T_Last[idx] = 1550.0;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx + 1];
	//		T_Right = T_New[idx - ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx - D];
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == (nx - 1) && m == 0)  //17
	//	{
	//		//T_Last[idx] = 1550.0;
	//		T_Up = T_New[idx - 1];
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx - ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx + D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j == (ny - 1) && i == (nx - 1) && m == (nz - 1))  //18
	//	{
	//		//T_Last[idx] = 1550.0;
	//		T_Up = T_New[idx - 1];
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx - ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx - D];
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //19
	//	{
	//		//T_Last[idx] = T_Cast;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx + ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx + D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //20
	//	{
	//		//T_Last[idx] = T_Cast;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx + ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx - D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == 0 && m == 0) //21
	//	{
	//		//T_Last[idx] = T_Cast;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx + 1];
	//		T_Right = T_New[idx + ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx + D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == 0)  //22
	//	{
	//		//T_Last[idx] = T_Cast;
	//		T_Up = T_New[idx - 1];
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx + ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx + D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == 0 && m == (nz - 1)) //23
	//	{
	//		//T_Last[idx] = T_Cast;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx + 1];
	//		T_Right = T_New[idx + ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx - D];
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == (nz - 1)) //24
	//	{
	//		//T_Last[idx] = T_Cast;
	//		T_Up = T_New[idx - 1];
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx + ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx - D];
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == 0 && m != 0 && m != (nz - 1))  //25
	//	{
	//		//T_Last[idx] = T_Cast;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx + 1] - 2 * dx * h * (T_New[idx] - Tw) / lamd;
	//		T_Right = T_New[idx + ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //26
	//	{
	//		//T_Last[idx] = T_Cast;
	//		T_Up = T_New[idx - 1] - 2 * dx * h * (T_New[idx] - Tw) / lamd;
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx + ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}

	//	else  //27
	//	{
	//		//T_Last[idx] = T_Cast;
	//		T_Up = T_New[idx + 1];
	//		T_Down = T_New[idx - 1];
	//		T_Right = T_New[idx + ND];
	//		T_Left = T_New[idx - ND];
	//		T_Forw = T_New[idx + D];
	//		T_Back = T_New[idx - D];
	//		T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
	//			+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
	//	}
	//}
}

int main()
{
	const int nx = 21, ny = 3000, nz = 21;   // nx is the number of grid in x direction, ny is the number of grid in y direction.
	//int num_blocksx = 1, num_blocksy = 1, num_blocksz = 1, num_threadsx = 1, num_threadsy = 1, num_threadsz = 1;// block number(1D)  thread number in x and y dimension(2D)
	int tnpts = 10000;  // time step
	float T_Cast = 1558.0, Lx = 0.25, Ly = 28.599, Lz = 0.25, t_final = 2000.0, dx, dy, dz, tao;  // T_Cast is the casting temperature Lx and Ly is the thick and length of steel billets
	float *T_Init;
	float ccml[Section + 1] = { 0.0,0.2,0.4,0.6,0.8,1.0925,2.27,4.29,5.831,9.6065,13.6090,19.87014,28.599 }; // The cooling sections
	float H_Init[Section] = { 1380,1170,980,800,1223.16,735.05,424.32,392.83,328.94,281.64,246.16,160.96 };  // The heat transfer coefficients in the cooling sections

	T_Init = (float *)calloc(nx * ny * nz, sizeof(float));  // Initial condition

	/*num_threadsx = 7;
	num_threadsy = 20;
	num_threadsz = 7;
	num_blocksx = nx/num_blocksx;
	num_blocksy = nx / num_blocksy;
	num_blocksz = nx / num_blocksz;*/

	for (int m = 0; m < nz; m++)
		for (int j = 0; j < ny; j++)
	       for (int i = 0; i < nx; i++)
			   T_Init[nx * ny * m + j * nx + i] = T_Cast;  // give the initial condition

	dx = Lx / (nx - 1);            // the grid size x
	dy = Ly / (ny - 1);            // the grid size y
	dz = Lz / (nz - 1);            // the grid size y
	tao = t_final / (tnpts - 1);   // the time step size
	//gridcheck(dx, dy, tao);

	printf("Casting Temperature = %f ", T_Cast);
	printf("\n");
	printf("The thick of steel billets(m) = %f ", Lx);
	printf("\n");
	printf("The length of steel billets(m) = %f ", Ly);
	printf("\n");
	printf("The length of steel billets(m) = %f ", Lz);
	printf("\n");
	printf("dx(m) = %f ", dx);
	printf("dy(m) = %f ", dy);
	printf("dz(m) = %f ", dz);
	printf("tao(s) = %f ", tao);
	printf("\n");
	printf("simulation time(s) = %f\n ", t_final);

	clock_t timestart = clock();
	cudaError_t cudaStatus = addWithCuda(T_Init, dx, dy, dz, tao, nx, ny, nz, tnpts, ccml, H_Init );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	clock_t timeend = clock();

	printf("running time = %d(millisecond)", (timeend - timestart));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

cudaError_t addWithCuda(float *T_Init, float dx, float dy, float dz, float tao, int nx, int ny, int nz, int tnpts, float *ccml, float *H_Init)
{
	float *dev_T_New, *dev_T_Last, *dev_ccml, *dev_H_Init; // the point on GPU
	float *T_Result;
	const int Num_Iter = 2000;                         // The result can be obtained by every Num_Iter time step
	volatile bool dstOut = true;
	FILE *fp = NULL;

	T_Result = (float *)calloc(nx * ny * nz, sizeof(float)); // The temperature of steel billets

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	HANDLE_ERROR(cudaSetDevice(0));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_New, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_T_Last, nx * ny * nz * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_ccml, (Section + 1) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_H_Init, Section * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_T_Last, T_Init, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_ccml, ccml, (Section + 1) * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_H_Init, H_Init, Section * sizeof(float), cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(num_threadsx, num_threadsy,num_threadsz);
	dim3 BlocksPerGrid(num_blocksx,num_blocksy,num_blocksz);
	// Launch a kernel on the GPU with one thread for each element.
	for (int i = 0; i < tnpts; i++)
	{
		addKernel << <BlocksPerGrid, threadsPerBlock >> >(dev_T_New, dev_T_Last, dev_ccml, dev_H_Init, dx, dy, dz, tao, nx, ny, nz, dstOut);
		dstOut = !dstOut;

		if (i % Num_Iter == 0) {
			HANDLE_ERROR(cudaMemcpy(T_Result, dev_T_Last, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
			printf("time_step = %d\n  simulation time is %f\n", i, i*tao);
			printf("%f, %f, %f, %f", T_Result[0], T_Result[(nx - 1)*(ny - 1)*(nz - 1) - nx], T_Result[(nx - 1)*(ny - 1)*(nz - 1) - nx * nz],T_Result[(nx - 1)*(ny - 1)*(nz - 1)]);
			printf("\n");
		}
	}

	fp = fopen("F:\\Temperature3DGPU_shared_memmory_Static.txt", "w");
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			for (int m = 0; m < nz; m++)
				fprintf(fp, " %f", T_Result[ nx * nz * j  + i * nz + m]);
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	// Check for any errors launching the kernel
	HANDLE_ERROR(cudaGetLastError());

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

__device__ float Boundary_Condition(int j, float dy, float *ccml_zone, float *H_Init)
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