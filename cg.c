#include <stdio.h>

#ifndef ITERATIONS
#define ITERATIONS 4 // number of stiffness matrices
#endif

#ifndef H
#define H 4 // stiffness matrix dimension
#endif

#ifndef N
#define N 1 // number of stiffness matrices
#endif

float vector_dot(float *vt,float *v){	
	float ts= 0.0f;	
	for (int i=0; i<H; i++) {
		ts+= vt[i]*v[i];
	}
	return ts;
}

// M*v function
void matrix_vector(float *m,float *v, float *t){
	#pragma unroll	
	for (int j=0; j<H; j++) {	
		for (int i=0; i<H; i++) {
		t[j]+= m[j*H+i]*v[i];		
		}	
	}	
}

// k*v function
void scalar_vector(float s,float *v, float *t){
	for (int i=0; i<H;i++) {
		t[i]= v[i]*s;
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
	for (int i=0; i<H;i++) {
		t[i]= v1[i]+v2[i];
	}
}


void main(void) {


	// input_data
	
	float X[H*N];
     	float A[H*H*N];
  	float B[H*N];
  
	// populating A's
	for (int k=0; k<N; k++) {     			// matrix index
		for(int j=0; j<H; j++) {		// row index
			for(int i=0; i<H; i++) {	// column index
				A[k*H*H + j*H + i] = 40786.00;
			}
    		}
	}

	// populating B's and initialializing X's
   	for (int k=0; k<N; k++) {			// vector indices
		for(int j=0; j<H; j++) {		// element index
			B[k*H + j] = 23.345; 
			X[k*H + j] = 0;
		}
	}


printf("Matrices: AX = B\n");
for (int j=0;j<H;j++) {
	for (int i=0;i<H;i++) {
		printf("%f ",A[j*H + i]);
	}
	printf(" %f  %f", X[j], B[j]);
	printf("\n");
}

	int iters = ITERATIONS;
	float X_local[H];
	for (int i=0; i<H;i++) 
		{ X_local[i]= 0; } 					// x = {0}
	float A_local[H*H];
	for (int i=0; i<(H*H);i++) 
		{ A_local[i]= A[i]; }		 			// local_copy of A
	float r[H];
	for (int i=0; i<H;i++) 
		{ r[i]= B[i]; }						// r = b		
	float rtr = vector_dot(r,r);					// rtr = r'r
	float p[H];
	for (int i=0; i<H;i++) 
		{ p[i]= r[i]; }						// p = r	
	float alpha, beta, rtrold; 
 	float Ap[H], alpha_p[H], alpha_Ap[H], beta_p[H] ;
	

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

	for (int i=0; i<H;i++) { X[i]= X_local[i]; }		 	//write result

	printf("AX = B ??\n");
	for (int j=0;j<H;j++) {
		for (int i=0;i<H;i++) {
			printf("%f ",A[j*H + i]);
		}
		printf(" %f  %f", X[j], B[j]);
		printf("\n");
	}
}
