#define H 4096 // Matrix size
#define BLOCK_SIZE 64 // Block size 
#define NNZ 52104 // Non zero entries
#define ME 13

#pragma OPENCL EXTENSION cl_khr_fp64 : enable


__kernel 
void matrix_vector (__global float *restrict Ap, __global int *restrict col, __global float *restrict val , __global float *restrict p) {				// kernel 2 (ND Range) ELL

	int id1 = get_global_id(1);
    	int id0 = get_global_id(0); 

		float running_sum= 0.0f;	
		
		#pragma unroll ME
		for (int j= 13; j> 0; j--) {
			int c = col[(id1*H*ME)+(id0*ME)+j-1];
			running_sum += val[(id1*H*ME)+(id0*ME)+j-1]*p[(id1*H)+c-1];
		}

		Ap[(id1*H)+id0] = running_sum;
}

__kernel	
__attribute((reqd_work_group_size(BLOCK_SIZE,1,1)))																		
void dot_product( __global float *restrict Ap, __global float *restrict p, __global float *restrict result ) {								// kernel 1 dot_product
	
    // Get the index of the current element
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);

    int lid0 = get_local_id(0);		// 0 to BLOCK_SIZE-1
    int lid1 = get_local_id(1);		// = 1

    int groupid0 = get_group_id(0);	
    int groupid1 = get_group_id(1);    	// problem id

	__local float product[BLOCK_SIZE];
	
	product[lid0] = p[(groupid1*H) + (groupid0*BLOCK_SIZE) +lid0] * Ap[(groupid1*H) + (groupid0*BLOCK_SIZE) +lid0];
	//complete work group
	barrier (CLK_LOCAL_MEM_FENCE);

	result[(id1*H)+id0] = product[lid0]; 
}

__kernel
void get_alpha( __global float *restrict result, __global float *restrict rtr, __global float *restrict alpha) {							// kernel 3 (Task)

	int id = get_global_id(0);
	float temp = 0.0f;
	float f = 1e-16;	
	
	#pragma unroll 1
	for (int i = 0; i<H; i++) {
		temp += result[(id*H)+i] ;
	}
	
	alpha[id] = rtr[id]/(temp+f);

}

__kernel
void get_beta( __global float *restrict result, __global float *restrict rtr, __global float *restrict beta) {								// kernel 5 (Task)
	
	int id = get_global_id(0);
    	float rtrnew = 0.0f;
    	float f = 1e-16;

	# pragma unroll 1	
	for (int k=0; k< H; k++){
		rtrnew += result[(id*H)+k];
	}
    beta[id] = rtrnew/(rtr[id]+f); 	
    rtr[id] = rtrnew;  
}


__kernel
__attribute((reqd_work_group_size(BLOCK_SIZE,1,1)))
void update_x_r( __global float *restrict Ap, __global float *restrict r, __global float *restrict p, __global float *restrict alpha, __global float *restrict x) {	// kernel 4 (ND_Range) (Work Groups)

    // Get the index of the current element
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);

    int lid0 = get_local_id(0);		// 0 to BLOCK_SIZE-1
    int lid1 = get_local_id(1);		// = 1

    int groupid0 = get_group_id(0);	
    int groupid1 = get_group_id(1);    	// problem id

	__local float partial_x[BLOCK_SIZE];
	__local float partial_r[BLOCK_SIZE];

	partial_x[lid0] = x[(groupid1*H) + (groupid0*BLOCK_SIZE) +lid0] + alpha[groupid1] * p[(groupid1*H) + (groupid0*BLOCK_SIZE) +lid0];
	partial_r[lid0] = r[(groupid1*H) + (groupid0*BLOCK_SIZE) +lid0] - alpha[groupid1] * Ap[(groupid1*H) + (groupid0*BLOCK_SIZE) +lid0];
	
	//complete work group
	barrier (CLK_LOCAL_MEM_FENCE);

	x[(id1*H)+id0] = partial_x[lid0];
	r[(id1*H)+id0] = partial_r[lid0];
	
}


__kernel
__attribute((reqd_work_group_size(BLOCK_SIZE,1,1)))
void update_p( __global float *restrict r, __global float *restrict p, __global float *restrict beta) {									// kernel 6 (ND_Range) (Work Groups)

    // Get the index of the current element
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);

    int lid0 = get_local_id(0);		// 0 to BLOCK_SIZE-1
    int lid1 = get_local_id(1);		// = 1

    int groupid0 = get_group_id(0);	
    int groupid1 = get_group_id(1);    	// problem id

	__local float partial[BLOCK_SIZE];

	partial[lid0] = r[(groupid1*H) + (groupid0*BLOCK_SIZE) +lid0] + beta[groupid1] * p[(groupid1*H) + (groupid0*BLOCK_SIZE) +lid0];
    	
	//complete work group
	barrier (CLK_LOCAL_MEM_FENCE);
	
	p[(id1*H)+id0]= partial[lid0];
}

