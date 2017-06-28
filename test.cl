#define H 4096 // Matrix size
#define BLOCK_SIZE 64 // Block size 

#pragma OPENCL EXTENSION cl_altera_channels : enable

channel float Ap0_ch;
channel float Ap1_ch;
channel float r0_ch;
channel float r1_ch;
channel float p0_ch;
channel float p1_ch;

__kernel
void dot_product( __global float *restrict r, __global float *restrict rtr) {			// kernel 1

	float prod = 0.0f;

	# pragma unroll 1	
	for (int k=0; k< H; k++){
		prod += r[k] * r[k];
	}	
	
	rtr[0]= prod;
}

__kernel // write to channel: Ap0, p0
__attribute((reqd_work_group_size(BLOCK_SIZE,1,1)))
void matrix_vector ( __global float *restrict A, __global float *restrict P) {			// kernel 2

    // Local storage for a block of input matrices A and P
    __local float A_local[BLOCK_SIZE][BLOCK_SIZE];
    __local float P_local[BLOCK_SIZE];				//changed

    // Block index
    int group_id = get_group_id(0);				//

    // Local ID index (offset within a block)
    int local_id = get_local_id(0);				//

    // Compute loop bounds
    int a_start = H * BLOCK_SIZE * group_id;			//
    int a_end   = a_start + H - 1;				//
    int p_start = 0;						//

    float running_sum = 0.0f;					// running sum for one work item (private memory)

    // Compute the matrix multiplication result for this output element. Each
    // loop iteration processes one block of the matrix.
    for (int a = a_start, p = p_start; a <= a_end; a += BLOCK_SIZE, p += BLOCK_SIZE )
    {
        // Load the matrices to local memory
	#pragma unroll 
	for (int i= 0; i<BLOCK_SIZE; i++){
	       	A_local[local_id][i] = A[a + H * local_id + i];	// try to improve !
	}
        P_local[local_id] = P[p + local_id];			//
	
        // Wait for the entire block to be loaded.
        barrier(CLK_LOCAL_MEM_FENCE);				 
	// one block loaded in local memory which will be accessed by BLOCK_SIZE work items


        // Do the dot product accumulation within this block. Fully unroll the loop.
        #pragma unroll 
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            running_sum += A_local[local_id][k] * P_local[k];	// accumulated sum for a block
        }
	// this sum keeps incrementing for subsequent blocks corresponding to a particular work item	
	
        // Wait for the block to be fully consumed before loading the next
        // block.
        barrier(CLK_LOCAL_MEM_FENCE);
	// local memory fully utilized by BLOCK_SIZE work items to produce corresponding running sums
    }

    // Store result in matrix C
    //AP[get_global_id(0)] = running_sum;
	write_channel_intel(Ap0_ch, running_sum);
	write_channel_intel(p0_ch, p[get_global_id(0)]);
	
}


__kernel // read from channel Ap0, p0 ; write to channel : p1, Ap1
void get_alpha( __global float *restrict rtr, __global float *restrict alpha ) {			// kernel 3

    float pAp = 0.0f;
    float Ap_in, p_in;
	
	# pragma unroll 1	
	for (int k=0; k< H; k++){
		Ap_in = read_channel_intel(Ap0_ch);
		p_in = read_channel_intel(p0_ch);
		pAp += p[k] * Ap;
		//pAp += p[k] * Ap[k];
		write_channel_intel(Ap1_ch, Ap);
		write_channel_intel(p1_ch, p[k]);
	}
	alpha[0] = rtr[0]/pAp;
}


__kernel	// read from channel : p1, Ap1 ; write to channel : p2, r0 
void update_x_r( __global float *restrict r, __global float *restrict x, __global float *restrict alpha) {			// kernel 4

	float Ap_in, p_in;
    // Get the index of the current element
    int j = get_global_id(0);
    
	Ap_in = read_channel_intel(Ap1_ch);
	p_in = read_channel_intel(p1_ch);
	x[j] = x[j] + alpha[0] * p;
	r_in = r[j] - alpha[0] * Ap;  
	write_channel_intel(p2_ch, p_in);
	write_channel_intel(r0_ch, r_in);
}


__kernel	// read from channel : r0 ; write to channel : r1
void get_beta( __global float *restrict rtr, __global float *restrict beta ) {							// kernel 5

    float rtrnew = 0.0f;    	
    float r_in;

	# pragma unroll 1	
	for (int k=0; k< H; k++){
		r_in = read_channel_intel(r0_ch);
		rtrnew = rtrnew + r_in * r_in;
		write_channel_intel(r1_ch, r_in);
	}
    
    	beta[0] = rtrnew/rtr[0];	
   	rtr[0] = rtrnew;  
}

__kernel	// read from channel : r1, p2
void update_p( __global float *restrict r, __global float *restrict p, __global float *restrict beta) {				// kernel 6

	float r_in, p_in;

    // Get the index of the current element
    int j = get_global_id(0);

	r_in = read_channel_intel(r1_ch);
	p_in = read_channel_intel(p2_ch);
    	p[j] = r_in + beta[0] * p_in; 
	r[j] = r_in; 	
}

