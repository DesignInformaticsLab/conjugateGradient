#ifndef H
#define H 64 // default value
#endif

//////////////////////////////////////////////////////////////////CONJUGATE GRADIENT

// m'*m function
float vector_dot(float *vt,float *v){	
	float ts= 0.0f;	
	#pragma unroll
	for (int i=0; i<H; i++) {
		ts+= vt[i]*v[i];
	}
	return ts;
}

// M*v function
void matrix_vector(float *m,float *v, float *t){
	#pragma unroll	
	for (int j=0; j<H; j++) {
		#pragma unroll		
		for (int i=0; i<H; i++) {
		t[j]+= m[j*H+i]*v[i];		
		}	
	}	
}

// k*v function
void scalar_vector(float s,float *v, float *t){
	#pragma unroll
	for (int i=0; i<H;i++) {
		t[i]*= s;
	}
}

// v1-v2 function
void vector_sub(float *v1, float *v2, float *t){
	for (int i=0; i<H;i++) {
		t[i]= v1[i]-v2[i];
	}
}

// v1+v2 function
void vector_add(float *v1, float *v2, float *t){
	#pragma unroll
	for (int i=0; i<H;i++) {
		t[i]= v1[i]+v2[i];
	}
}

__kernel void cg(__global float *restrict X, __global float *restrict A, __global float *restrict B, int A_width, int B_width) {
	
	int iters = 100;
	float *X_local = {0};
	#pragma unroll
	for (int i=0; i<H;i++) { X_local[i]= 0; } 			// x = {0}
	float A_local[H*H];
	#pragma unroll
	for (int i=0; i<(H*H);i++) { A_local[i]= A[i]; } 		// local_copy of A
	float *r = {0};
		#pragma unroll
		for (int i=0; i<H;i++) { r[i]= B[i]; }			// r = b		
	float rtr = vector_dot(r,r);					// rtr = r'r
	float *p = {0};
		#pragma unroll
		for (int i=0; i<H;i++) { p[i]= r[i]; }			// p = r	
	float alpha, beta, rtrold; 
 	float *Ap= {0}, *alpha_p= {0}, *alpha_Ap= {0}, *beta_p={0} ;
	
	#pragma unroll
	for (int k=0; k<iters; k++) {

		matrix_vector(A_local,p, Ap);
		alpha= rtr/vector_dot(p,Ap);				// alpha
		
		scalar_vector(alpha,p,alpha_p);				
		vector_add(X_local,alpha_p,X_local);			// update x

		scalar_vector(alpha,Ap,alpha_Ap);     		
		vector_sub(r,alpha_Ap,r);				// update r
    		
		rtrold = rtr;						
    		rtr = vector_dot(r,r);
    		beta = rtr / rtrold;					// beta
		
		scalar_vector(beta,p,beta_p);
    		vector_add(r,beta_p,p);					// update p	
		}

	#pragma unroll
	for (int i=0; i<H;i++) { X[i]= X_local[i]; } 			//write result
}

