#include <assert.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#ifndef H
#define H 20 // stiffness matrix dimension
#endif

#ifndef N
#define N 8 // number of stiffness matrices
#endif

using namespace aocl_utils;
using namespace std;

int main(void) {

  cl_int status;
  if(!setCwdToExeDir()) {
    return false;
  }

// runtime configurations
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;


  // Get the OpenCL platform.								// platform
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL devices.						// device
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;
  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  // We'll just use the first device.
  device = devices[0];

  // Create the context.								// context
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queue.								// command_queue
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Create the program.								// program
  std::string binary_file = getBoardBinaryFile("cg", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the		// kernel
  // original CL file, that was compiled into an AOCX file using the AOC tool
  const char *kernel_name = "cg";  // Kernel name, as defined in the CL file
  kernel = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create kernel");

// problem data // to be read from text files later!					// input_data
	
	float *X = new float[H*N];
     	float *A = new float[H*H*N];
  	float *B = new float[H*N];
  
	// populating A's
	for (int k=0; k<N; k++) {     			// matrix index
		for(int j=0; j<H; j++) {		// row index
			for(int i=0; i<H; i++) {	// column index
				A[k*H*H + j*H + i] = 2.00;
			}
    		}
	}

	// populating B's and initialializing X's
   	for (int k=0; k<N; k++) {			// vector indices
		for(int j=0; j<H; j++) {		// element index
			B[k*H + j] = 1.00; 
			X[k*H + j] = 0;
		}
	}

    // Create memory buffers on the device for each matrix
    cl_mem x_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*sizeof(float), NULL, &status);
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*H*sizeof(float), NULL, &status);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*sizeof(float), NULL, &status);

    // Copy the matrix A, B and X to each device memory counterpart
    status = clEnqueueWriteBuffer(queue, x_mem_obj, CL_TRUE, 0, N*H*sizeof(float), X, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, a_mem_obj, CL_TRUE, 0, N*H*H*sizeof(float), A, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, b_mem_obj, CL_TRUE, 0, N*H*sizeof(float), B, 0, NULL, NULL);

  // Set the kernel arguments
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &x_mem_obj);		
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &a_mem_obj);		
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &b_mem_obj);		
  checkError(status, "Failed to set kernel arguments");

  int dimention = 1; 					// an array (1-D) of matrices (work-items)
  size_t gSize = N;
  // Launch the kernel
  status = clEnqueueNDRangeKernel(queue, kernel, dimention, NULL, N, , 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  // Wait for command queue to complete pending events
  status = clFinish(queue);
  checkError(status, "Failed to finish");

  printf("\nKernel execution is complete.\n");

// Clean up
    status =clReleaseKernel(kernel);
    status = clReleaseProgram(program);
    status = clReleaseMemObject(a_mem_obj);
    status = clReleaseMemObject(b_mem_obj);
    status = clReleaseMemObject(x_mem_obj);
    status = clReleaseCommandQueue(queue);
    status = clReleaseContext(context);
    return 0;

}

