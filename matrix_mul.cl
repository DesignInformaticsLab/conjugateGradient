#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1 // default value 64
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 64 // default value
#endif

///////////////////////////////////////////////////////NAIVE MATRIX VECTOR MULTIPLICATION

__kernel void matrix_mul( __global float *restrict C, __global float *A, __global float *B, int A_width, int B_width) {
    
    // Get the index of the current element
    //int size = get_global_size(0);
    int x = get_global_size(0);
    int y = get_global_size(1);
 
    int i = get_global_id(0);
    int j = get_global_id(1);

    float acc = 0.0f;

	if (i < x && j < y) {
    	for (int k=0; k< A_width; k++)
	    	acc += A[j*A_width + k] * B[k*x + i];
		C[j*x + i] = acc;
	}
}

//////////////////////////////////////////////////////TILED MATRIX MULTIPLICATION


__kernel void matrix_mul_tile(__global float *restrict C, __global float *A, __global float *B, int A_width, int B_width) {
    int size = get_global_size(0);
    const int l_i = get_local_id(0); // Local col ID (max: TILE_SIZE)
    const int l_j = get_local_id(1); // Local row ID (max: TILE_SIZE)
    const int g_i = get_global_id(0); // Global col ID of C (0..SIZE)
    const int g_j = get_global_id(1); // Global row ID of C (0..SIZE)
 
    // Local memory to fit a tile of TILE_SIZE*TILE_SIZE elements of A and B
    __local int Asub[TILE_SIZE][TILE_SIZE];
    __local int Bsub[TILE_SIZE][TILE_SIZE];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    int t_i;
    int t_j;
    // Loop over all tiles
    const int numTiles = size/TILE_SIZE;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        t_i = t*TILE_SIZE + l_i;
        t_j = t*TILE_SIZE + l_j;

        Asub[l_i][l_j] = A[g_j*size + t_i];
        Bsub[l_i][l_j] = B[t_j*size + g_i];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TILE_SIZE; k++) {
            acc += Asub[k][l_j] * Bsub[l_i][k];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    C[g_j*size + g_i] = acc;
   
}
