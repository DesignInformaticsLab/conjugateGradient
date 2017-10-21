#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <fstream>
#include <iostream>

#define AOCL_ALIGNMENT 64

#define ITERATIONS 1 // iterations

#define H 4096 // stiffness matrix dimension
#define BLOCK_SIZE 64 // Block size
#define NNZ 52104 // Non zero entries

#define N 4096 // number of stiffness matrices

using namespace std;
using namespace aocl_utils;

// runtime configurations
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel1 = NULL, kernel2 = NULL, kernel3 = NULL, kernel4 = NULL, kernel5 = NULL, kernel6 = NULL;
static cl_program program = NULL;

// Function prototypes
bool init();
void cleanup();

// Entry point
int main(void) {

  int i,j,k;
  float f;
  int max_elements=0;
  int row_number=0;
  cl_int status;

  if(!init()) {
    return -1;
  }

  
  // problem data from text files
	
	float *X = NULL;
	posix_memalign ((void**)&X, AOCL_ALIGNMENT, H*N*sizeof(float));
  	float *B = NULL;
	posix_memalign ((void**)&B, AOCL_ALIGNMENT, H*N*sizeof(float));
	float *rtr = NULL;
	posix_memalign ((void**)&rtr, AOCL_ALIGNMENT, N*sizeof(float));

//////////////////////////////////////////////////////////////////////////////////////
	ifstream inFileA;
	inFileA.open("/home/user/sanimesh/data.txt");

  	if (!inFileA) {
  	  cout << "Cannot open file A\n";
  	}

//initializing
	for (k=0; k<N; k++) {
		for (j = 0; j < H; j++) {
		X[k*H + j] = 0.00;
		B[k*H + j] = 1.00;
		}	
	rtr[k] = H;
	}

//	load to temporary memory
     	int *row_temp = new int[N*NNZ];
	int *row_first = new int[N*NNZ];
	int *col_temp = new int[N*NNZ];
	int *col_first = new int[N*NNZ];
	float *val_temp = new float[N*NNZ];
	float *val_first = new float[N*NNZ];
	int *row = new int[N*(H+1)];

	// stiffness matrices
	int n=0;
	while(!inFileA.eof()) {
		inFileA >> row_temp[n];
		inFileA >> col_temp[n];
		inFileA >> val_temp[n];
		n++;
	}

//////////////////////////////////////////////////////////////CSR Modified

// sort row indices
k=0;
for ( i=0; i< H; i++) {
	for ( j=0; j<NNZ; j++) {
		if (row_temp[j]==i+1) { row_first[k]=row_temp[j]; col_first[k]=col_temp[j]; val_first[k]=val_temp[j]; k++;}
	}
}

k=0;
for ( i=0; i< H; i++) {
	for ( j=0; j<NNZ; j++) {
		if (row_first[j]==i+1) { row[i]=j; break; }
	}
}
row[H]=NNZ;

// find max_elements
for ( i=0; i< H; i++) {
	k=0;
	for ( j=0; j<NNZ; j++) {
		if (row_first[j]==i+1) k++;
	}
	if (k > max_elements){max_elements = k;row_number = i+1 ;}
}



////////////////////////////////////////////////////////////////////////////////////// Modified CSR format	
     	int *col = NULL;
	posix_memalign ((void**)&col, AOCL_ALIGNMENT, H*max_elements*N*sizeof(int));
  	float *val = NULL;
	posix_memalign ((void**)&val, AOCL_ALIGNMENT, H*max_elements*N*sizeof(float));

for (i=0; i< (H*max_elements); i++) {
		col[i]= 1;
		val[i]= 0.00;
}

int count;
int pointer = 0;

for (k=0; k<H; k++) {
	count = row[k+1] - row[k] ;
	for (i=0; i<count; i++) {
		col[k*max_elements + i]= col_first[pointer];
		val[k*max_elements + i]= val_first[pointer];
		pointer++;
	}
}
cout << pointer << endl;

// multiple probelms data copies
	
		for (k=1; k<N; k++) {
			for (i=0; i<(H*max_elements); i++) {
				col[(k*H*max_elements)+i]=col[i];
				val[(k*H*max_elements)+i]=val[i];
			}
		}	

///////////////////////////////////////////////////////////////

inFileA.close();



    // Create memory buffers on the device for each matrix
    cl_mem x_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*sizeof(float), NULL, &status);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*sizeof(float), NULL, &status);
    cl_mem r_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*sizeof(float), NULL, &status);
    cl_mem p_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*sizeof(float), NULL, &status);
    cl_mem rtr_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &status);
    cl_mem alpha_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &status);
    cl_mem beta_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*sizeof(float), NULL, &status);
    cl_mem Ap_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*sizeof(float), NULL, &status);
    cl_mem beta_p_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*sizeof(float), NULL, &status);
    cl_mem result_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*sizeof(float), NULL, &status);

    	cl_mem col_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*max_elements*sizeof(int), NULL, &status);
    	cl_mem val_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, N*H*max_elements*sizeof(float), NULL, &status);

    
    // Copy the matrix A, B and X to each device memory counterpart
    status = clEnqueueWriteBuffer(queue, x_mem_obj, CL_TRUE, 0, N*H*sizeof(float), X, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, b_mem_obj, CL_TRUE, 0, N*H*sizeof(float), B, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, r_mem_obj, CL_TRUE, 0, N*H*sizeof(float), B, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, p_mem_obj, CL_TRUE, 0, N*H*sizeof(float), B, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, rtr_mem_obj, CL_TRUE, 0, N*sizeof(float), rtr, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, alpha_mem_obj, CL_TRUE, 0, N*sizeof(float), rtr, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, beta_mem_obj, CL_TRUE, 0, N*sizeof(float), rtr, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, Ap_mem_obj, CL_TRUE, 0, N*H*sizeof(float), X, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, beta_p_mem_obj, CL_TRUE, 0, N*H*sizeof(float), X, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, result_mem_obj, CL_TRUE, 0, N*H*sizeof(float), X, 0, NULL, NULL);

    	status = clEnqueueWriteBuffer(queue, col_mem_obj, CL_TRUE, 0, N*H*max_elements*sizeof(int), col, 0, NULL, NULL);
    	status = clEnqueueWriteBuffer(queue, val_mem_obj, CL_TRUE, 0, N*H*max_elements*sizeof(float), val, 0, NULL, NULL);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  int dimention = 1; 					// (H work-items)
  int dimention2 = 2;
  size_t global_item_size[3] = {H,1,1} ;
  size_t local_item_size[3] = {BLOCK_SIZE,1,1} ;
  size_t global_item_size2[3] = {H,N,1} ;
  size_t local_item_size2[3] = {BLOCK_SIZE,1,1} ;
  size_t global_item_size_spmv[3] = {NNZ,1,1} ;		// spmv	
  size_t local_item_size_spmv[3] = {NNZ,1,1} ;
  size_t global_size[3] = {N,1,1};			// multiple probelms
  size_t local_size[3] = {N,1,1};

float time = getCurrentTimestamp();

for (i = 0; i < ITERATIONS; i++) {

// Set the kernel2 arguments
  status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *) &Ap_mem_obj);					// this !
  checkError(status, "Failed to set kernel2 argument0");		
  status = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *) &col_mem_obj);		
  checkError(status, "Failed to set kernel2 argument1");
  status = clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void *) &val_mem_obj);		
  checkError(status, "Failed to set kernel2 argument2");
  status = clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void *) &p_mem_obj);		
  checkError(status, "Failed to set kernel2 argument3");

  // Launch the kernel2
  status = clEnqueueNDRangeKernel(queue, kernel2, dimention2, NULL, global_item_size2, local_item_size2, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel2");

// Set the kernel1 arguments
  status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *) &Ap_mem_obj);					
  checkError(status, "Failed to set kernel1 argument0");		
  status = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *) &p_mem_obj);		
  checkError(status, "Failed to set kernel1 argument1");  
  status = clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *) &result_mem_obj);		
  checkError(status, "Failed to set kernel1 argument2");  

  // Launch the kernel1										
  status = clEnqueueNDRangeKernel(queue, kernel1, dimention2, NULL, global_item_size2  , local_item_size2, 0, NULL, NULL);	// no work group
  checkError(status, "Failed to launch kernel1");

  // Set the kernel3 arguments
  status = clSetKernelArg(kernel3, 0, sizeof(cl_mem), (void *) &result_mem_obj);
  checkError(status, "Failed to set kernel3 argument0");		
  status = clSetKernelArg(kernel3, 1, sizeof(cl_mem), (void *) &rtr_mem_obj);		
  checkError(status, "Failed to set kernel3 argument1");
  status = clSetKernelArg(kernel3, 2, sizeof(cl_mem), (void *) &alpha_mem_obj);		
  checkError(status, "Failed to set kernel3 argument2");
  
  // Launch the kernel3
  //status = clEnqueueTask(queue, kernel3, 0, NULL, NULL);
  status = clEnqueueNDRangeKernel(queue, kernel3, dimention, NULL, global_size, local_size, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel3");

    // Set the kernel4 arguments
  status = clSetKernelArg(kernel4, 0, sizeof(cl_mem), (void *) &Ap_mem_obj);
  checkError(status, "Failed to set kernel4 argument0");		
  status = clSetKernelArg(kernel4, 1, sizeof(cl_mem), (void *) &r_mem_obj);		
  checkError(status, "Failed to set kernel4 argument1");
  status = clSetKernelArg(kernel4, 2, sizeof(cl_mem), (void *) &p_mem_obj);		
  checkError(status, "Failed to set kernel4 argument2");
  status = clSetKernelArg(kernel4, 3, sizeof(cl_mem), (void *) &alpha_mem_obj);		
  checkError(status, "Failed to set kernel4 argument3");
  status = clSetKernelArg(kernel4, 4, sizeof(cl_mem), (void *) &x_mem_obj);		
  checkError(status, "Failed to set kernel4 argument4");

  // Launch the kernel4
  status = clEnqueueNDRangeKernel(queue, kernel4, dimention2, NULL, global_item_size2 , local_item_size2, 0, NULL, NULL);
  //status = clEnqueueTask(queue, kernel4, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel4");

// Set the kernel1 arguments
  status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *) &r_mem_obj);					
  checkError(status, "Failed to set kernel1 argument0");		
  status = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *) &r_mem_obj);		
  checkError(status, "Failed to set kernel1 argument1");  
  status = clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *) &result_mem_obj);		
  checkError(status, "Failed to set kernel1 argument2");  

  // Launch the kernel1										
  status = clEnqueueNDRangeKernel(queue, kernel1, dimention2, NULL, global_item_size2  , local_item_size2, 0, NULL, NULL);	// no work group
  checkError(status, "Failed to launch kernel1"); 

  // Set the kernel5 arguments
  status = clSetKernelArg(kernel5, 0, sizeof(cl_mem), (void *) &result_mem_obj);
  checkError(status, "Failed to set kernel5 argument0");		
  status = clSetKernelArg(kernel5, 1, sizeof(cl_mem), (void *) &rtr_mem_obj);		
  checkError(status, "Failed to set kernel5 argument1");
  status = clSetKernelArg(kernel5, 2, sizeof(cl_mem), (void *) &beta_mem_obj);		
  checkError(status, "Failed to set kernel5 argument2");
  
  // Launch the kernel5
 // status = clEnqueueTask(queue, kernel5, 0, NULL, NULL);
  status = clEnqueueNDRangeKernel(queue, kernel5, dimention, NULL, global_size, local_size, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel5");

  // Set the kernel6 arguments
  status = clSetKernelArg(kernel6, 0, sizeof(cl_mem), (void *) &r_mem_obj);
  checkError(status, "Failed to set kernel6 argument0");		
  status = clSetKernelArg(kernel6, 1, sizeof(cl_mem), (void *) &p_mem_obj);		
  checkError(status, "Failed to set kernel6 argument1");
  status = clSetKernelArg(kernel6, 2, sizeof(cl_mem), (void *) &beta_mem_obj);		
  checkError(status, "Failed to set kernel6 argument2");		

  // Launch the kernel6
  status = clEnqueueNDRangeKernel(queue, kernel6, dimention2, NULL, global_item_size2  , local_item_size2, 0, NULL, NULL);	// no work group
  //status = clEnqueueTask(queue, kernel6, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel6");

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Wait for command queue to complete pending events
  status = clFinish(queue);
  checkError(status, "Failed to finish");

time = getCurrentTimestamp() - time;	

  // Read the output
  status = clEnqueueReadBuffer(queue, x_mem_obj, CL_TRUE, 0, N*H*sizeof(float), X, 0, NULL, NULL);
  checkError(status, "Failed to read output");
  status = clEnqueueReadBuffer(queue, rtr_mem_obj, CL_TRUE, 0, N*sizeof(float), rtr, 0, NULL, NULL);
  checkError(status, "Failed to read rtr"); 


//cout << "maximum elements in a row is "<< max_elements << endl;
//cout << "row containing maximum elements is "<< row_number << endl;


for (j=0; j<8; j++)	{
	cout << "	"<< rtr[j] ;
}

cout << endl;

//for (j=0;j<26;j++) { cout << col[j] << "	" << val[j] << "	" << endl ;}

for (j=0;j<20;j++) {
	//cout << (j+1) << "	"<< row [j] << "	"<< col[j]<< "	" << val[j] << endl ;															// debug
	cout << (j+1) << "	"<< X [j] << "	"<< X [H+j] <<"	"<< X [(H*2) + j]<<"	"<< X [(H*3) + j] << "	"<< X [(H*4)+j] << "	"<< X [(H*5) + j] << "	"<< X [(H*6)+j] << "	"<< X [(H*7)+j] <<endl;	// multiple problem
	//cout << (j+1) << "	"<< X [j] << endl;																				// 1 problem
}


printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));

    cleanup();
    clReleaseMemObject(x_mem_obj);
    clReleaseMemObject(b_mem_obj);
    clReleaseMemObject(r_mem_obj);
    clReleaseMemObject(p_mem_obj);
    clReleaseMemObject(rtr_mem_obj);
    clReleaseMemObject(alpha_mem_obj);
    clReleaseMemObject(beta_mem_obj);
    clReleaseMemObject(beta_p_mem_obj);
    clReleaseMemObject(Ap_mem_obj);

   	clReleaseMemObject(col_mem_obj);
    	clReleaseMemObject(val_mem_obj); 

    free(X);
    free(B);
    free(rtr);	
    free(col);
    free(val);
    
	return 0;

}

bool init() {
  cl_int status;
  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.								// platform
  platform = findPlatform("Altera");
  
  if(platform == NULL) {
    printf("ERROR: Unable to find OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL devices.						// device
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;
  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  // We'll just use the first device.
  device = devices[1];

  // Create the context.								// context
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queues.								// command_queues
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
  kernel1 = clCreateKernel(program, "dot_product", &status);
  checkError(status, "Failed to create kernel1");
  kernel2 = clCreateKernel(program, "matrix_vector", &status);
  checkError(status, "Failed to create kernel2");
  kernel3 = clCreateKernel(program, "get_alpha", &status);
  checkError(status, "Failed to create kernel3");
  kernel4 = clCreateKernel(program, "update_x_r", &status);
  checkError(status, "Failed to create kernel4");
  kernel5 = clCreateKernel(program, "get_beta", &status);
  checkError(status, "Failed to create kernel5");
  kernel6 = clCreateKernel(program, "update_p", &status);
  checkError(status, "Failed to create kernel6");

return true;
}

void cleanup() { 
  if(kernel1)
    clReleaseKernel(kernel1);
  if(kernel2)
    clReleaseKernel(kernel2);  
  if(kernel3)
    clReleaseKernel(kernel3);  
  if(kernel4)
    clReleaseKernel(kernel4);  
  if(kernel5)
    clReleaseKernel(kernel5);  
  if(kernel6)
    clReleaseKernel(kernel6);
  if(program)
    clReleaseProgram(program);
  if(queue)
    clReleaseCommandQueue(queue);
  if(context)
    clReleaseContext(context);
}
