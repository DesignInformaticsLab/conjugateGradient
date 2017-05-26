#include <stdio.h>

#ifndef ITERATIONS
#define ITERATIONS 123 // number of stiffness matrices
#endif

#ifndef H
#define H 20 // stiffness matrix dimension
#endif

#ifndef N
#define N 1 // number of stiffness matrices
#endif

float vector_dot(float *vt,float *v){	
	float ts= 0.0f;	
	int i;
	for (i=0; i<H; i++) {
		ts+= vt[i]*v[i];
	}
	return ts;
}

// M*v function
void matrix_vector(float *m,float *v, float *t){
	int i,j;
	for (j=0; j<H; j++) {
		t[j]=0;	
		for (i=0; i<H; i++) {
		t[j]+= m[j*H+i]*v[i];		
		}	
	}	
}

// k*v function
void scalar_vector(float s,float *v, float *t){
	int i;	
	for (i=0; i<H;i++) {
		t[i]= v[i]*s;
	}
}

// v1-v2 function
void vector_sub(float *v1, float *v2, float *t){
	int i;
	for (i=0; i<H;i++) {
		t[i]= v1[i]-v2[i];
	}
}

// v1+v2 function
void vector_add(float *v1, float *v2, float *t){
	int i;
	for (i=0; i<H;i++) {
		t[i]= v1[i]+v2[i];
	}
}


void main(void) {

	int i,j,k;
	const float f = 1e-16;
	printf("value of constant f %f \n", f);
	// input_data
	
	float X[H*N];
     	float A[H*H*N];
  	float B[H*N];

	FILE *inFile;
	inFile = fopen("stiffness1.txt", "r");

  	if (!inFile) {
  	  printf("Cannot open file.\n");
  	}

	// populating A's
	for (k = 0; k < N; k++) {
		for (j = 0; j < H; j++) {
			for (i=0; i<H; i++) {
				fscanf(inFile,"%f",&A[k*H*H + j*H + i]);
			}
		}
	}
		
	fclose(inFile);

	// populating A's
/*	for (int k=0; k<N; k++) {     			// matrix index
		for(int j=0; j<H; j++) {		// row index
			for(int i=0; i<H; i++) {	// column index
				A[k*H*H + j*H + i] = 40786.00;
			}
    		}
	}
*/

	// populating B's and initialializing X's
   	for (k=0; k<N; k++) {			// vector indices
		for(j=0; j<H; j++) {		// element index
			B[k*H + j] = 1.00; 
			X[k*H + j] = 0.00;
		}
	}
	


printf("Matrices: AX = B\n");
for (j=0;j<H;j++) {
	for (i=0;i<H;i++) {
		printf("%f ",A[j*H + i]);
	}
	printf(" %f  %f", X[j], B[j]);
	printf("\n");
}

	int iters = ITERATIONS;
	float X_local[H];
	for (i=0; i<H;i++) 
		{ X_local[i]= 0; } 					// x = {0}
	float A_local[H*H];
	for (i=0; i<(H*H);i++) 
		{ A_local[i]= A[i]; }		 			// local_copy of A
	float r[H];
	for (i=0; i<H;i++) 
		{ r[i]= B[i]; }						// r = b		
	float rtr = vector_dot(r,r);					// rtr = r'r
	float p[H];
	for (i=0; i<H;i++) 
		{ p[i]= r[i]; }						// p = r	
	float alpha, beta, rtrold, pAp; 
 	float Ap[H], alpha_p[H], alpha_Ap[H], beta_p[H] ;
	
	for (k=0; k<iters; k++) {

		printf("iteration: %d \n", (k+1));		
		printf("rtr_old= %f \n",rtr);	
		matrix_vector(A_local,p, Ap);				//Ap
		
		printf("Ap:\n");
                for (j=0;j<H;j++) {
                        printf("%f\n",Ap[j]);
                }

		pAp= vector_dot(p,Ap);
		printf("pAp= %f \n",pAp);				//pAp
		alpha= rtr/(pAp+f);				
		printf("alpha= %f \n", alpha);				// alpha

		scalar_vector(alpha,p,alpha_p);
		printf("alpha_p:\n");
                for (j=0;j<H;j++) {
                        printf("%f\n",alpha_p[j]);
                }

						
		vector_add(X_local,alpha_p,X_local);			// update x
		printf("X:\n");
                for (j=0;j<H;j++) {
                        printf("%f\n",X_local[j]);
                }

		
		scalar_vector(alpha,Ap,alpha_Ap);     		
		printf("alpha_Ap:\n");
                for (j=0;j<H;j++) {
                        printf("%f\n",alpha_Ap[j]);
                }


		vector_sub(r,alpha_Ap,r);				// update r

		printf("residual:\n");
		for (j=0;j<H;j++) {
			printf("%f\n",r[j]);
		}

		rtrold = rtr;				
    		rtr = vector_dot(r,r);
		printf("rtr_new= %f \n",rtr);

    		beta = rtr / (rtrold+f);					// beta
		printf("beta= %f \n", beta);
		
		scalar_vector(beta,p,beta_p);
		printf("beta_p:\n");
                for (j=0;j<H;j++) {
                        printf("%f\n",beta_p[j]);
                }

    		vector_add(r,beta_p,p);					// update p	
		printf("p:\n");
                for (j=0;j<H;j++) {
                        printf("%f\n",p[j]);
                }

		for (i=0; i<H;i++) { X[i]= X_local[i]; }		 //write result
	}
}
