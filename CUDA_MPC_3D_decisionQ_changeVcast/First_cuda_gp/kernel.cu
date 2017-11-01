//在MPC模型中将h改为Q,拉速改变1.2m/min到1.0m/min
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_occupancy.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include "book.h"
#include "gridcheck.h"
using namespace std;
# define Section 12  // number of cooling sections
# define CoolSection 8
# define MoldSection 4
# define StaticIter 50
# define M 3
# define N M+2*CoolSection
# define TestIter 1000
# define limit 1001//limit>=tnpts/num_iter

float ccml[Section + 1] = { 0.0,0.2,0.4,0.6,0.8,1.0925,2.27,4.29,5.831,9.6065,13.6090,19.87014,28.599 }; // The cooling sections
float H_Init[Section] = { 1380,1170,980,800,1223.16,735.05,424.32,392.83,328.94,281.64,246.16,160.96 };  // The heat transfer coefficients in the cooling sections
//float H_Init[Section] = { 1400,1200,1000,800,1200,750,400,400,350,300,250,150 };
//float H_Init[Section] = { 1500,1300,1100,900,1300,850,500,500,450,400,350,250 };
//float H_Init_Temp[Section] = { 1380,1170,980,800,1223.16,735.05,424.32,392.83,328.94,281.64,246.16,160.96 };  // The heat transfer coefficients in the cooling sections
float H_Init_Temp[Section] = { 0 };
float H_Init_Final[Section] = { 1380 };
float Q_air[CoolSection] = { 200,1500,850,650,1000,850,400,480 };
float Taim[CoolSection] = { 966.149841, 925.864746, 952.322083, 932.175537, 914.607117, 890.494263, 870.804443, 890.595825 };
float delta_z[Section] = {2.7,2.7,1.8,1.8,1.8,1.8,1.8,0.9};
float *Calculation_MeanTemperature(int nx, int ny, int nz, float dy, float *ccml, float *T, float num);
float *calculateThickness(float *T_result, int nx, int ny, int nz, float dy, float *ccml, float Ts, float thick);
cudaError_t addWithCuda(float *T_Init, float dx, float dy, float dz, float tao, int nx, int ny, int nz, int tnpts, int num_blocks, int num_threadsx, int num_threadsy);
__device__ void Physicial_Parameters(float T, float *pho, float *Ce, float *lamd);
__device__ float Boundary_Condition(int j, float dx, float *ccml_zone, float *H_Init);
float *relationshiphandQ(float *h_Init, float* Q_air);
float stop_criterion();
float update_c(float[], float c0,int iter);
void update_lamda(float[],int iter,float[]);
float alfa[limit] = { 1.0 };
float g[N] = { 0 };
float testArray[TestIter] = { 0 };

__global__ void addKernel(float *T_New, float *T_Last, float *ccml, float *H_Init, float dx, float dy, float dz, float tao, int nx, int ny, int nz, bool disout,float Vcast)
{
	int i = threadIdx.x;
	int m = threadIdx.y;
	int j = blockIdx.x;
	int idx = j * nx * nz + m * nx + i;
	int ND = nx * nz;
	int D = nx;

	float pho, Ce, lamd; // physical parameters pho represents desity, Ce is specific heat and lamd is thermal conductivity
	float a, T_Up, T_Down, T_Right, T_Left, T_Forw, T_Back, h = 100.0, Tw = 30.0, T_Cast = 1558.0; //Vcast = -0.02

	if (disout) {
		Physicial_Parameters(T_Last[idx], &pho, &Ce, &lamd);
		a = (lamd) / (pho*Ce);
		h = Boundary_Condition(j, dy, ccml, H_Init);
		if (j == 0) //1
		{
			T_New[idx] = T_Cast;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //10
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = (a*tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ (a*tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m != 0 && m != (nz - 1)) //11
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //12
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //13
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //14
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == 0)  //15
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == (nz - 1))  //16
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == 0)  //17
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == (nz - 1))  //18
		{
			//T_New[idx] = 1550.0;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx - ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //19
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //20
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == 0) //21
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == 0)  //22
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx + D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == (nz - 1)) //23
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == (nz - 1)) //24
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx - 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx - D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m != 0 && m != (nz - 1))  //25
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx + 1] - 2 * dx * h * (T_Last[idx] - Tw) / lamd;
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //26
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx - 1] - 2 * dx * h * (T_Last[idx] - Tw) / lamd;
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else  //27
		{
			//T_New[idx] = T_Cast;
			T_Up = T_Last[idx + 1];
			T_Down = T_Last[idx - 1];
			T_Right = T_Last[idx + ND];
			T_Left = T_Last[idx - ND];
			T_Forw = T_Last[idx + D];
			T_Back = T_Last[idx - D];
			T_New[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_Last[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}
	}

	else
	{
		Physicial_Parameters(T_New[idx], &pho, &Ce, &lamd);
		a = (lamd) / (pho*Ce);
		h = Boundary_Condition(j, dy, ccml, H_Init);
		if (j == 0) //1
		{
			T_Last[idx] = T_Cast;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m != 0 && m != (nz - 1)) //10
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m != 0 && m != (nz - 1)) //11
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //12
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //13
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //14
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == 0)  //15
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == 0 && m == (nz - 1))  //16
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == 0)  //17
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j == (ny - 1) && i == (nx - 1) && m == (nz - 1))  //18
		{
			//T_Last[idx] = 1550.0;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx - ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == 0)  //19
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i != 0 && i != (nx - 1) && m == (nz - 1))  //20
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D] - 2 * dz * h * (T_Last[idx] - Tw) / lamd;
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == 0) //21
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == 0)  //22
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx + D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m == (nz - 1)) //23
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m == (nz - 1)) //24
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx - 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx - D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == 0 && m != 0 && m != (nz - 1))  //25
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx + 1] - 2 * dx * h * (T_New[idx] - Tw) / lamd;
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else if (j != 0 && j != (ny - 1) && i == (nx - 1) && m != 0 && m != (nz - 1)) //26
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx - 1] - 2 * dx * h * (T_New[idx] - Tw) / lamd;
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}

		else  //27
		{
			//T_Last[idx] = T_Cast;
			T_Up = T_New[idx + 1];
			T_Down = T_New[idx - 1];
			T_Right = T_New[idx + ND];
			T_Left = T_New[idx - ND];
			T_Forw = T_New[idx + D];
			T_Back = T_New[idx - D];
			T_Last[idx] = a*(tao / (dx*dx))*T_Up + a*(tao / (dx*dx))*T_Down + ((1 - 2 * a*tao / (dx*dx) - 2 * a*tao / (dy*dy) - 2 * a*tao / (dz*dz) + tao*Vcast / dy))*T_New[idx]
				+ a*(tao / (dy*dy))*T_Right + (a*tao / (dy*dy) - tao*Vcast / dy)*T_Left + (a*tao / (dz*dz))*T_Forw + (a*tao / (dz*dz))*T_Back;
		}
	}
}
int main()
{
	const int nx = 21, ny = 3000, nz = 21;   // nx is the number of grid in x direction, ny is the number of grid in y direction.
	int num_blocks = 1, num_threadsx = 1, num_threadsy = 1;// num_threadsz = 1; // block number(1D)  thread number in x and y dimension(2D)
	int tnpts = 10001;  // time step
	float T_Cast = 1558.0, Lx = 0.25, Ly = 28.599, Lz = 0.25, t_final = 2000.0, dx, dy, dz, tao;  // T_Cast is the casting temperature Lx and Ly is the thick and length of steel billets
	
	float *T_Init;
	num_threadsx = nx;
	num_threadsy = nz;
	num_blocks = ny;

	T_Init = (float*)calloc(nx*ny*nz,sizeof(float));  // Initial condition

	//for (int m = 0; m < nz; m++)
	//	for (int j = 0; j < ny; j++)
	//       for (int i = 0; i < nx; i++)
	//		   T_Init[nx * ny * m + j * nx + i] = T_Cast;  // give the initial condition
	

    //读取txt文件
	ifstream in("F:\\Temperature3DGPU_shared_memmory_Static.txt");
	if (!in)
	{
		cerr << "open the filename failed!" << endl;
		return 1;
	}
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			for (int m = 0; m < nz; m++)
				in >> T_Init[nx * nz * j + i * nz + m];
		}
	}
	in.close();
	ofstream fout;
	fout.open("F:\\data_zf\\testTcastTinit.txt");
	if (!fout)
		cout << "testTcastTinit is not open" << endl;
	else
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				for (int m = 0; m < nz; m++)
					fout << T_Init[nx * nz * j + i * nz + m] << ", ";
				fout << endl;
			}
			fout << endl;
		}
	}
	fout.close();

	dx = Lx / (nx - 1);            // the grid size x
	dy = Ly / (ny - 1);            // the grid size y
	dz = Lz / (nz - 1);            // the grid size z
	tao = t_final / (tnpts - 1);   // the time step size
	//gridcheck(dx, dy, tao);

	cout << "Casting Temperature " << T_Cast << endl;
	cout << "The length of steel billets(m) " << Ly << endl;
	cout << "The width of steel billets(m) " << Lz << endl;
	cout << "The thick of steel billets(m) " << Lx << endl;
	cout << "dx(m) " << dx << ", ";
	cout << "dy(m) " << dy << ", ";
	cout << "dz(m) " << dz << ", ";
	cout << "tao(s) " << tao << ", ";
	cout << "simulation time(s) " << t_final << endl;

	//clock_t timestart = clock();

	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	cudaError_t cudaStatus = addWithCuda(T_Init, dx, dy, dz, tao, nx, ny, nz, tnpts, num_blocks, num_threadsx, num_threadsy);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsetime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsetime, start, stop));
	cout << "running time =" << (elapsetime);

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	/*clock_t timeend = clock();

	cout << "running time = " << (timeend - timestart);*/

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

cudaError_t addWithCuda(float *T_Init, float dx, float dy, float dz, float tao, int nx, int ny, int nz, int tnpts, int num_blocks, int num_threadsx, int num_threadsy)
{
	float *dev_T_New, *dev_T_Last, *dev_ccml, *dev_H_Init; // the point on GPU
	float *T_Result, *Delta_H_Init, *T_HoldLast,*ThickAll, **Mean_TSurfaceElement, **Mean_TSurfaceElementOne;
	float *Point_TSurfaceElement, *Point_TSurfaceElementOne, **Mean_TCenterElement, **Mean_TCenterElementOne;
	float **JacobianMatrix, *JacobianG0, *JacobianG1, *JacobianG2, *TZ_gradient,*partionQ;
	float **JacobinTZgradient, **TZ_gradientElement, **TZ_gradientElementOne;
	float dh = 10.0,dQ=1.0, arf1, arf2, step = -0.0001,T_bmax=1100,Ts=1462,Tl= 1518.0,Tu=-100,Td=200;
	float Vcast = -0.02;
	const int Num_Iter = 10, PrintLabel = 0;// The result can be obtained by every Num_Iter time step
	volatile bool dstOut = true;

	//约束函数
	float c[limit] = {10};
	float norm_g[limit] = { 0 };
	float eps = 0.0001,c0 = 10;	
	float lamda[limit][N] = { 1 };
	float gtest[limit][N] = { 0 };
	float htest[limit][Section] = { 0 };
	float fitness[limit] = { 0 };
	float gfitness[limit] = { 0 };

	T_Result = (float *)calloc(nx * ny * nz, sizeof(float)); // The temperature of steel billets
	Delta_H_Init = (float*)calloc(CoolSection, sizeof(float));
	T_HoldLast = (float*)calloc(nz * ny * nx, sizeof(float));
	Point_TSurfaceElement = (float*)calloc(CoolSection, sizeof(float));
	Point_TSurfaceElementOne = (float*)calloc(CoolSection, sizeof(float));
	JacobianG0 = (float*)calloc(CoolSection, sizeof(float));
	JacobianG1 = (float*)calloc(CoolSection, sizeof(float));
	JacobianG2 = (float*)calloc(CoolSection, sizeof(float));
	TZ_gradient = (float*)calloc(CoolSection, sizeof(float));
	partionQ = (float*)calloc(CoolSection, sizeof(float));
	ThickAll = (float*)calloc(Section, sizeof(float));

	JacobianMatrix = (float**)calloc(CoolSection, sizeof(float));
	for (int i = 0; i < CoolSection; i++)
		JacobianMatrix[i] = (float*)calloc(CoolSection, sizeof(float));
	
	Mean_TSurfaceElement = (float**)calloc(CoolSection, sizeof(float));
	for (int i = 0; i < CoolSection; i++)	
		Mean_TSurfaceElement[i] = (float*)calloc(CoolSection, sizeof(float));	

	Mean_TSurfaceElementOne = (float**)calloc(CoolSection, sizeof(float));
	for (int i = 0; i < CoolSection; i++)
		Mean_TSurfaceElementOne[i] = (float*)calloc(CoolSection, sizeof(float));

	Mean_TCenterElement = (float**)calloc(CoolSection, sizeof(float));
	for (int i = 0; i < CoolSection; i++)
		Mean_TCenterElement[i] = (float*)calloc(CoolSection, sizeof(float));

	Mean_TCenterElementOne = (float**)calloc(CoolSection, sizeof(float));
	for (int i = 0; i < CoolSection; i++)
		Mean_TCenterElementOne[i] = (float*)calloc(CoolSection, sizeof(float));

	JacobinTZgradient = (float**)calloc(CoolSection, sizeof(float));
	for (int i = 0; i < CoolSection; i++) 
		JacobinTZgradient[i] = (float*)calloc(CoolSection, sizeof(float));
	
	TZ_gradientElement = (float**)calloc(CoolSection, sizeof(float));
	for (int i = 0; i < CoolSection; i++)
		TZ_gradientElement[i] = (float*)calloc(CoolSection, sizeof(float));

	TZ_gradientElementOne = (float**)calloc(CoolSection, sizeof(float));
	for (int i = 0; i < CoolSection; i++)
		TZ_gradientElementOne[i] = (float*)calloc(CoolSection, sizeof(float));

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

	dim3 threadsPerBlock(num_threadsx, num_threadsy);
	float SurfaceError[TestIter / 10+1][CoolSection];
	for (int t = 0; t < TestIter*10+1; t++)
	{			
		if (t / Num_Iter >= 8*StaticIter)//400-500
			Vcast = -0.017;
		//else if (t / Num_Iter >= (10*StaticIter + 100) && t / Num_Iter < (10*StaticIter + 200))//600-700
		//	Vcast = -0.02;
		//else if (t / Num_Iter >= (10*StaticIter + 200) && t / Num_Iter < (10*StaticIter + 300))//700-800
		//	Vcast = -0.023;
		/*else
			Vcast = -0.02;*/
		if (t % Num_Iter == 0)
		{
			int iter = t / Num_Iter;
			HANDLE_ERROR(cudaMemcpy(T_HoldLast, dev_T_Last, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
			for (int m = 0; m < CoolSection + 1; m++)
			{			
				if (m == CoolSection)
				{
					for (int temp = 0; temp < Section; temp++) {
						H_Init_Temp[temp] = H_Init[temp];
					}
						
					HANDLE_ERROR(cudaMemcpy(dev_H_Init, H_Init_Temp, Section * sizeof(float), cudaMemcpyHostToDevice));
					for (int PNum = 0; PNum < Num_Iter; PNum++)
					{
						addKernel << <num_blocks, threadsPerBlock >> >(dev_T_New, dev_T_Last, dev_ccml, dev_H_Init, dx, dy, dz, tao, nx, ny, nz, dstOut,Vcast);
						dstOut = !dstOut;
					}

					HANDLE_ERROR(cudaMemcpy(T_Result, dev_T_New, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
					float* Mean_TSurface = Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result,0);  // calculation the mean surface temperature of steel billets in every cooling sections
					float* Mean_TPoint = Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result, 8.0 / 250 * nx);//一个点的温度
					float Point_TSurface = Mean_TPoint[MoldSection];
					float *Mean_TCenter = Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result,nx/2);//中心温度，为啥是中心呢？
					for (int temp = 0; temp < CoolSection; temp++) {
						
						Point_TSurfaceElementOne[temp] = Point_TSurface;
						if (iter >= StaticIter) 
						{
							if (temp < CoolSection - 1)
								TZ_gradient[temp] = (Mean_TSurface[temp + 1 + MoldSection] - Mean_TSurface[temp + MoldSection]) / delta_z[temp];
							else
								//TZ_gradient[temp] = -(T_Result[nx*nz*(ny - 1) + 0 * nz + (int)(nx - 1)] - Mean_TSurface[temp + MoldSection]) / delta_z[temp];	
								TZ_gradient[temp] = 100;
							//printf("TZ_gradient=%f  ", TZ_gradient[temp]);
						}
						for (int column = 0; column < CoolSection; column++) 
						{
							Mean_TSurfaceElementOne[temp][column] = Mean_TSurface[column + MoldSection];
							Mean_TCenterElementOne[temp][column] = Mean_TCenter[column + MoldSection];
							TZ_gradientElementOne[temp][column] = TZ_gradient[column + MoldSection];
						}
					}
					//printf("\n");
				}

				else
				{				

					for (int temp = 0; temp < Section; temp++) 						
						H_Init_Temp[temp] = H_Init[temp];			
					//printf("  h=%f", H_Init_Temp[m]);
					H_Init_Temp[m + MoldSection] = H_Init[m + MoldSection] + dh;
					HANDLE_ERROR(cudaMemcpy(dev_H_Init, H_Init_Temp, Section * sizeof(float), cudaMemcpyHostToDevice));

					for (int PNum = 0; PNum < Num_Iter; PNum++)//预测时段长度
					{
						addKernel << <num_blocks, threadsPerBlock >> >(dev_T_New, dev_T_Last, dev_ccml, dev_H_Init, dx, dy, dz, tao, nx, ny, nz, dstOut,Vcast);
						dstOut = !dstOut;
					}

					HANDLE_ERROR(cudaMemcpy(T_Result, dev_T_New, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
					float* Mean_TSurface = Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result,0); // calculation the mean surface temperature of steel billets in every cooling sections
					float* Mean_TPoint = Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result, 8.0 / 250 * nx);
					Point_TSurfaceElement[m] = Mean_TPoint[m];
					float *Mean_TCenter = Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result,nx/2);					
					if (iter >= 2 * StaticIter) 
					{
						if (m < CoolSection - 1)
							TZ_gradient[m] = (Mean_TSurface[m + 1 + MoldSection] - Mean_TSurface[m + MoldSection]) / delta_z[m];
						else
							//TZ_gradient[m] = -(T_Result[nx*nz*(ny - 1) + 0 * nz + (int)(nx - 1)] - Mean_TSurface[m + MoldSection]) / delta_z[m];
							TZ_gradient[m] = 150;
					}				
					for (int column = 0; column < CoolSection; column++)
					{						
						Mean_TSurfaceElement[m][column] = Mean_TSurface[column + MoldSection];//和二冷区对应
						Mean_TCenterElement[m][column] = Mean_TCenter[column + MoldSection];
						TZ_gradientElement[m][column] = TZ_gradient[column + MoldSection];
					}					
					/*if (iter >= StaticIter) {
						g[0] = Mean_TSurface[MoldSection] - T_bmax;
						g[1] = Mean_TPoint[MoldSection]-Ts;
						g[2] = Mean_TCenter[MoldSection + CoolSection - 2] - Tl;
					}*/
				}
				/*for (int i = 0; i < CoolSection; i++) {
					printf("TZ_gradient=%f  ",TZ_gradient[i]);
				}
				printf("\n");*/
				
				//添加约束,第一段二冷区的平均温度 
				if (iter >= StaticIter)
				{
					g[0] = Mean_TSurfaceElement[0][0] - T_bmax;
					g[2] = Mean_TCenterElement[MoldSection + 1][MoldSection + 1] - Tl;
					g[1] = Point_TSurfaceElement[MoldSection] - Ts;
					/*printf("g[0]=%f\n", g[0]);
					printf("g[1]=%f\n", g[1]);
					printf("g[2]=%f\n", g[2]);*/
					for (int i = M; i < N; i++) {
						if (i < M + CoolSection)
							g[i] = Tu - TZ_gradient[i-M];
						else
							g[i] = TZ_gradient[i - M - CoolSection]-Td;						
					}				
				}
				for (int temp = 0; temp < M; temp++)
					gfitness[iter] += lamda[iter][temp] * g[temp];
				for (int i = 0; i < N; i++)
				{
					if (iter < StaticIter)
						gtest[iter][i] = 0;
					else
						gtest[iter][i] = g[i];
				}			
				HANDLE_ERROR(cudaMemcpy(dev_T_Last, T_HoldLast, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice));

			}

			printf("iter=%d\n", iter);
			printf("g[0]=%f\n", g[0]);
			//判断是否跳出循环
			if (iter >= StaticIter)
			{
				norm_g[iter] = stop_criterion();
				norm_g[0] = norm_g[StaticIter];
			}
			if (norm_g[iter - 1]<eps&&iter>2*StaticIter)//满足停止准则
				break;

			//更新乘子			
			c[iter] = update_c(norm_g, c0, iter-StaticIter);
			printf("c=%f\n", c[iter]);

			if (iter <= StaticIter)
				for (int j = 0; j < N; j++) 
					lamda[iter][j] = 1;
			/*for (int j = 0; j < N; j++) {
				printf("lamda[i]=%f\n", lamda[iter][j]);
				printf("g[i]=%f\n", g[j]);
			}*/

			for (int j = 0; j < Section; j++) {
				htest[iter][j] = H_Init_Temp[j];
			}
			//目标函数梯度
			for (int row = 0; row < CoolSection; row++)
			{
				for (int column = 0; column < CoolSection; column++)
				{
					JacobianMatrix[row][column] = (Mean_TSurfaceElement[row][column] - Mean_TSurfaceElementOne[row][column]) / dh;	//复合导数1
					JacobinTZgradient[row][column] = (TZ_gradientElement[row][column] - TZ_gradientElementOne[row][column]) / dh;
					if (row == 0)
					{
						if (iter > StaticIter)
						{
							JacobianG0[column] = (Mean_TSurfaceElement[0][column] - Mean_TSurfaceElementOne[0][column]) / dh;
							JacobianG1[column] = (Point_TSurfaceElement[column]- Point_TSurfaceElementOne[column]) / dh;
							JacobianG2[column] = (Mean_TCenterElement[4][column] - Mean_TCenterElementOne[4][column]) / dh;
						}
					}
				}
			}				


			for (int temp = 0; temp < CoolSection; temp++) {
				Delta_H_Init[temp] = 0.0;
				for (int column = 0; column < CoolSection; column++)
				{
					Delta_H_Init[temp] += (Mean_TSurfaceElementOne[temp][column] - Taim[column]) * JacobianMatrix[temp][column];//复合导数2
					if (iter > StaticIter)
					{
						Delta_H_Init[temp] += lamda[iter][temp + M] * JacobinTZgradient[temp][column];
						Delta_H_Init[temp] +=(-1)* lamda[iter][temp + M + CoolSection] * JacobinTZgradient[temp][column];
					}					
					
				}
				Delta_H_Init[temp] += H_Init[temp] - H_Init_Final[temp];//增加的h的增量部分
				gfitness[iter] += lamda[iter][temp + M] * g[temp + M];
				gfitness[iter] += lamda[iter][temp + M + CoolSection] * g[temp + M + CoolSection];
				fitness[iter]+= pow(H_Init[temp] - H_Init_Final[temp],2);
				gfitness[iter] += pow(H_Init[temp] - H_Init_Final[temp], 2);

				
				if (iter > StaticIter)
				{
					Delta_H_Init[temp] += lamda[iter][0] * JacobianG0[temp];//增广形式的导数部分1
					Delta_H_Init[temp] += lamda[iter][1] * JacobianG1[temp];//增广形式的导数部分2
					Delta_H_Init[temp] += lamda[iter][2] * JacobianG2[temp];//增广形式的导数部分3

				}
				//printf("  Delta_H_Init=%f\n", Delta_H_Init[temp]);
			}

			printf("\n");



			//步长根据目标函数改变
			arf1 = 0.0, arf2 = 0.0;
			for (int temp = 0; temp < CoolSection; temp++)
			{
				for (int column = 0; column < CoolSection; column++)
				{
					arf1 += ((Mean_TSurfaceElementOne[0][temp] - Taim[temp]) * JacobianMatrix[temp][column]) * Delta_H_Init[column];
					if (iter > StaticIter)
					{
						arf1 += (lamda[iter][temp + M] * JacobinTZgradient[temp][column] * Delta_H_Init[column]);
						arf1 += (lamda[iter][temp + M + CoolSection] * (-1)*JacobinTZgradient[temp][column] * Delta_H_Init[column]);
					}
					arf2 += JacobianMatrix[temp][column] * Delta_H_Init[column] * JacobianMatrix[temp][column] * Delta_H_Init[column];
				}
				//arf1 += (H_Init[temp] - H_Init_Final[temp])*Delta_H_Init[temp];
				//if (iter > StaticIter)
				{
					arf1 += lamda[iter][0] * JacobianG0[temp] * Delta_H_Init[temp];//增广形式部分1
					arf1 += lamda[iter][1] * JacobianG1[temp] * Delta_H_Init[temp];//增广形式部分2
					arf1 += lamda[iter][2] * JacobianG2[temp] * Delta_H_Init[temp];//增广形式部分2
				}
			}
			step = -arf1 / ((arf2)+0.001);//步长公式跟随目标函数改变,为啥要加0.001？			
			testArray[iter] = step;
			printf("step=%f\n", step);

			//迭代过程
			for (int temp = 0; temp < CoolSection; temp++)
			{
				H_Init_Final[temp] = H_Init[temp];
				H_Init[temp + MoldSection] += step *Delta_H_Init[temp];			
				//printf("  h=%f", H_Init[temp + MoldSection]);
			}			
			float *Q_water = relationshiphandQ(H_Init, Q_air);
			/*for (int temp = 0; temp < CoolSection; temp++) {
				printf("  Q_water=%f", Q_water[temp]*1000/60);
			}*/
			if (iter >= StaticIter)
			{
				for (int j = 0; j < N; j++)
				{
					lamda[iter + 1][j] = lamda[iter][j] + c[iter] * g[j];//lamda的更新
					if (lamda[iter + 1][j] < 0)
						lamda[iter + 1][j] = 0;//保证系数lamda的非负性
					if (lamda[iter + 1][j] > 100)
						lamda[iter][j] /= lamda[iter][j];
				}
			}
		}
		
			//实际模拟连铸过程
		for (int temp = 0; temp < Section; temp++)
		{
			H_Init_Temp[temp] = H_Init[temp];

		}
		     
			HANDLE_ERROR(cudaMemcpy(dev_H_Init, H_Init_Temp, Section * sizeof(float), cudaMemcpyHostToDevice));
			addKernel << <num_blocks, threadsPerBlock >> >(dev_T_New, dev_T_Last, dev_ccml, dev_H_Init, dx, dy, dz, tao, nx, ny, nz, dstOut,Vcast);
			dstOut = !dstOut;
			HANDLE_ERROR(cudaMemcpy(T_Result, dev_T_Last, nx * ny * nz* sizeof(float), cudaMemcpyDeviceToHost));
			float* Mean_TSurface = Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result, 0);  // calculation the mean surface temperature of steel billets in every cooling sections
			
			for (int temp = 0; temp < CoolSection; temp++) {
				fitness[t / Num_Iter] += pow((Mean_TSurface[temp + MoldSection] - Taim[temp]), 2);
				gfitness[t / Num_Iter] += pow((Mean_TSurface[temp + MoldSection] - Taim[temp]), 2);
			}
			
			if (t % (10 * Num_Iter) == 0)
			{				
				//结晶器的约束
                /*int thickness = 0;
				for (; thickness < nx / 2; thickness++) {
					float *Mean_Thickness = Calculation_MeanTemperature(nx, ny, nz, dy, ccml, T_Result, thickness);
					if (Mean_Thickness[MoldSection] > Ts)
						break;					
				}*/
				//printf("thickness=%d\n", thickness);
				

				ThickAll = calculateThickness(T_Result, nx, ny, nz, dy, ccml, Ts, 250);//厚度为0.25m				
				cout << endl<<"  ThickAll= " << endl;
				for (int temp = 0; temp < CoolSection; temp++)
					cout << ThickAll[temp + MoldSection] << ", ";
				

				cout << "  time_step = " << t << ",  " << "simulation time = " << t * tao;
				cout << endl << "TSurface = " << endl;
				for (int temp = 0; temp < CoolSection; temp++)
					cout << Mean_TSurface[temp + MoldSection] << ", ";

				cout << endl << "TSurface - Taim = " << endl;
				for (int temp = 0; temp < CoolSection; temp++)
				{
					cout << (Mean_TSurface[temp + MoldSection] - Taim[temp]) << ", ";
					SurfaceError[t / (10 * Num_Iter)][temp] = (Mean_TSurface[temp + MoldSection] - Taim[temp]);
				}
				cout << endl;
			}	
	}
	    
	ofstream fout;
		fout.open("F:\\data_zf\\changeVcastGPUMPC3D2block3threads.txt");
		if (!fout)
			cout << "changeVcastGPUMPC3D2block3threads is not open" << endl;
		else
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					for (int m = 0; m < nz; m++)
						fout << T_Result[nx * nz * j + i * nz + m] << ", ";
					fout << endl;
				}
				fout << endl;
			}
		}		
		fout.close();

		fout.open("F:\\data_zf\\changeVcastSurfaceGPUMPC3D2block3threads.txt");
		if (!fout)
			cout << "changeVcastSurfaceGPUMPC3D2block3threads is not open" << endl;
		else 
		{
			for (int j = 0; j < ny; j++)
			{
				fout << T_Result[nx * nz * j + 0 * nz + int((nx - 1) / 2)] << ", ";
			
				fout << endl;
			}
		}		
		fout.close();
	    
		fout.open("F:\\data_zf\\changeVcastCenterGPUMPC3D2block3threads.txt");
		if (!fout)
			cout << "changeVcastCenterGPUMPC3D2block3threads is not open" << endl;
		else
		{
			for (int j = 0; j < ny; j++)
			{
				fout << T_Result[nx * nz * j + int((nx - 1) / 2) * nz + int((nx - 1) / 2)] << ", ";

				fout << endl;
			}
		}
		fout.close();

		fout.open("F:\\data_zf\\changeVcastSurfaceErrorGPUMPC3D2block3threads.txt");
		if (!fout)
			cout << "changeVcastSurfaceErrorGPUMPC3D2block3threads is not open" << endl;
		else
		{
			for (int i = 0; i < TestIter / 10+1; i++)
			{
				for (int j = 0; j < CoolSection; j++)
					fout << SurfaceError[i][j] << ",";
				fout << endl;
			}
		}					
		fout.close();

		fout.open("F:\\data_zf\\changeVcastThicknessGPUMPC3D2block3threads.txt");
		if (!fout)
			cout << "changeVcastThicknessGPUMPC3D2block3threads is not open" << endl;
		else
		{
			for (int i = 0; i < CoolSection; i++) {
				fout << ThickAll[i + MoldSection] << ",";
				fout << endl;
			}
			
		}
		fout.close();

		fout.open("F:\\data_zf\\changeVcastlamda.txt");
		if (!fout)
			cout << "changeVcastlamda is not open" << endl;
		else
		{
			for (int i = 0; i < TestIter; i++)
			{
				for (int j = 0; j < N; j++)
					fout << lamda[i][j] << ",";
				fout << endl;			
			}
		}	
		fout.close();

		fout.open("F:\\data_zf\\changeVcastgtest.txt");
		if (!fout)
			cout << "changeVcastgtest is not open" << endl;
		else
		{
			for (int i = 0; i < TestIter; i++)
			{
				for (int j = 0; j < N; j++)
					fout << gtest[i][j] << ",";
				fout << endl;
			}
		}		
		fout.close();

		fout.open("F:\\data_zf\\changeVcasthtest.txt");
		if (!fout)
			cout << "changeVcasthtest is not open" << endl;
		else
		{
			for (int i = 0; i < TestIter; i++)
			{
				for (int j = 0; j < Section; j++)
					fout << htest[i][j] << ",";
				fout << endl;
			}
		}
		fout.close();

		fout.open("F:\\data_zf\\changeVcastfitnesstest.txt");
		if (!fout)
			cout << "changeVcastfitnesstest is not open" << endl;
		else
		{
			for (int i = 0; i < TestIter; i++)
			{
				fout << fitness[i] << ",";

				fout << endl;
			}
		}
		fout.close();

		fout.open("F:\\data_zf\\changeVcastgfitnesstest.txt");
		if (!fout)
			cout << "changeVcastgfitnesstest is not open" << endl;
		else
		{
			for (int i = 0; i < TestIter; i++)
			{
				fout << gfitness[i] << ",";

				fout << endl;
			}
		}
		fout.close();

		fout.open("F:\\data_zf\\changeVcastc.txt");
		if (!fout)
			cout << "changeVcastc is not open" << endl;
		else
		{
			for (int i = 0; i < TestIter; i++)
			{
				fout << c[i] << ",";

				fout << endl;
			}
		}							
		fout.close();

		fout.open("F:\\data_zf\\changeVcaststep.txt");
		if (!fout)
			cout << "changeVcaststep is not open" << endl;
		else
		{
			for (int i = 0; i < TestIter; i++)
			{
				fout << testArray[i] << ",";
				fout << endl;
			}
		}		
		fout.close();

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
	cudaFree(T_Init);
	cudaFree(dev_T_New);
	cudaFree(dev_T_Last);
	cudaFree(dev_ccml);
	cudaFree(dev_H_Init);
	cudaFree(JacobianMatrix);
	cudaFree(JacobianG0);
	cudaFree(JacobianG1);
	cudaFree(JacobianG2);
	cudaFree(JacobinTZgradient);

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

float* Calculation_MeanTemperature(int nx, int ny, int nz, float dy, float *ccml, float *T, float num)
{
	float y;
	int count = 0;
	int i = 0;

	float* Mean_Temperature;
	Mean_Temperature = new float[Section];
	for (int i = 0; i < Section; i++)
	{
		Mean_Temperature[i] = 0.0;
		for (int j = 0; j < ny - num; j++)
		{
			y = j * dy;
			if (y > *(ccml + i) && y <= *(ccml + i + 1))
			{
				Mean_Temperature[i] = Mean_Temperature[i] + T[nx * nz * j + (int)(num * nz) + int((nx - 1) / 2)];
				count++;
			}
		}
		Mean_Temperature[i] = Mean_Temperature[i] / float(count);
		count = 0;
	}
	return Mean_Temperature;
}
float stop_criterion() {
	float norm_g = 0.0;
	for (int i = 0; i <= N - 1; i++)
		norm_g = norm_g + g[i] * g[i];
	norm_g = sqrt(norm_g);
	//printf("norm_g=%f\n", norm_g);
	return(norm_g);
}

float update_c(float norm_g[], float c0,int iter) {//采用Luh论文中公式20
	float dM = 2.0, r = 0.5, p = 0.0, c = 10;
	if (iter > 0)
	{
		p = 1.0 - 1.0 / pow(iter, r);//p的更新公式67

		alfa[iter - 1] = 1.0 - 1.0 / (dM*pow(iter, p));//这个应该是iter代啊？？67
	}

	if (iter <= StaticIter)
		return c0;

	else
	{		
		/*for(int i=0;i<iter;i++)
		printf("  alfa=%f",alfa[i]);
		printf("\n");*/

		c = c0*norm_g[0] / norm_g[iter - 1];//c的迭代公式一部分20

		//printf("norm_g[0]=%f\n", norm_g[0]);
		//printf("norm_g[iter-1]=%f\n", norm_g[iter-1]);

		for (int i = 0; i <= iter - 1; i++)
			c = c*alfa[i];//c的迭代公式二部分20
	}

	return(c);
}
float *relationshiphandQ(float *h_Init, float* Q_air)
{
	float hx[CoolSection] = { 56.5,40.2,40.2,40.2,40.2,40.2,40.2,40.2 };
	float rw[CoolSection] = { 0.845,0.568,0.568,0.568,0.568,0.568,0.568,0.568 };
	float ra[CoolSection] = { 0.2,0.1902,0.1902,0.1902,0.1902,0.1902,0.1902,0.1902 };
	float hr[CoolSection] = { 0.15,0.082,0.082,0.082,0.082,0.082,0.082,0.082 };
	float Sw[CoolSection] = { 1.8,3.86,1.8,1.8, 1.8, 1.8, 1.8, 1.8 };
	float Sl[CoolSection] = { 0.5,0.8,2.5,1.8,4.0,3.5,6.0,8.9 };

	float *Q_water;
	Q_water = new float[CoolSection];
	for (int i = 0; i < CoolSection; i++)
	{
		Q_water[i] = pow((h_Init[i+MoldSection] - hr[i]) / hx[i] / pow(Q_air[i] / (Sl[i] * Sw[i]), ra[i]), 1 / rw[i])*(Sl[i] * Sw[i]);
	}
	return Q_water;
}
float *calculateThickness(float *T_result, int nx, int ny, int nz, float dy, float *ccml,float Ts,float thick) {
	float y;
	int count = 0;
	float *Mean_Temperature;
	Mean_Temperature = new float[Section];
	int *thickness;
	thickness = new int[Section];
	float *res_thickness;
	res_thickness = new float[Section];
	for (int i = 0; i < Section; i++)	
	{
		for (thickness[i] = 0; thickness[i] <nx; thickness[i]++){
			Mean_Temperature[i] = 0.0;
			for (int j = 0; j < ny; j++)
			{
				y = j * dy;
				if (y > *(ccml + i) && y <= *(ccml + i + 1))
				{
					Mean_Temperature[i] = Mean_Temperature[i] + T_result[nx * nz * j + thickness[i] * nz + int((nx - 1) / 2)];
					count++;
				}
			}
			Mean_Temperature[i] = Mean_Temperature[i] / float(count);
			count = 0;
			if (Mean_Temperature[i] > Ts)
				break;
	   }	
		res_thickness[i]=thickness[i] * thick / nx;
	}
	return res_thickness;
}
