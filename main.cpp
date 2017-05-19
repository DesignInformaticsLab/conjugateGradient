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
#define H 20 // default value
#endif

using namespace aocl_utils;
using namespace std;

// runtime configurations
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;


int main(void) {

// problem data
    float *A = new float[H*H];
    float *B = new float[H];
    float *X = new float[H];

    //Initialize matrix
    for(int j=0; j<SIZE; j++) {
		for(int i=0; i<SIZE; i++) {
			A[j*SIZE + i] = 1;
			B[j*SIZE + i] = i+1;
			C[j*SIZE + i] = 0;
			C_seq[j*SIZE + i] = 0;
		}
    }



