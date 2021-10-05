#import "MTSOS.h"
import math
import numpy as np
from scipy.linalg import _fblas as fblas
#include <math.h>
#include "MTSOS.h"
#include "mex.h"
#include "blas.h"
#/*a barrier for a two degree freedom space craft model with maximum permitted force*/

def so_barrier(S,S_prime,S_dprime,b, a, u, S_length,U_size,State_size,indicator,kappa,variables,variables_length,H_barrier,G_barrier,F_barrier,cnt):



    #F, G, and H are all passed in as NULL pointers;
    
    
    #------------
    
    #double *fx, *Df, *A;
    
    #char* uplo = "U";//used for the blas call to dsyr.
    

   

    A = [None] * 4#(S_length-1)
    alpha, C = 0,0
    uplo = "U"
    one,u_size = 0,0
    i = 0
    Df = []
    status = 1
    #A = []


    #------------------
    
    one = 1
    u_size = U_size
    #//pull the terms needed that are stored in variables;
    C = variables[1]
    # if cnt ==1:
    #     print((u))
    
    #//calculate the value of f(x)-c for a constraint f(x) < c;
    #fx = mxMalloc((S_length-1)*sizeof(double));
    fx = [0.0] * (S_length-1)
    fx2 = [0.0] * (S_length-1)
    if fx == None or fx2 == None:
        status = 0
    else:
    
        valid = 1 #;//a parameter to ensure that the inequality constraint is always met.

        for i in range(S_length-1):
            if valid == 1:
                fx[i] =  u[i*U_size]*u[i*U_size]+u[i*U_size+1]*u[i*U_size+1]-C
                fx2[i] = u[i*U_size]-(0.5)*C
                #print(fx[i])
                if ((fx[i] > 0) or (fx2[i] > 0)):
                    valid = 0

        
        if valid != 0:
            if indicator < 3:
                Df = [None] * (U_size*(S_length-1))
                if Df == None:
                    status = 0
                else:
                    for i in range(S_length-1):
                        Df[i*U_size] = 2*u[i*U_size]
                        Df[i*U_size+1] = 2*u[i*U_size+1]
                        #print(u[i*U_size], u[i*U_size+1])
                if indicator < 2 and status:
                    block_length = (2+U_size)*(2+U_size)
                    for i in range((2+U_size)*(2+U_size)*(S_length-1)):
                        H_barrier[i] = 0
                    A = [None] * 4
                    if A == None:
                        status = 0
                    else:
                        A[1] = 0
                    for i in range(S_length-1):
                        if status ==1:
                            alpha = 1.0/(fx[i]*fx[i])
                            A[0] = (-1)*2.0/fx[i]
                            A[2] = 0
                            A[3] = (-1)*2.0/fx[i]
                            A = np.array(A, np.float)
                            
                            temp_df = np.zeros((2,1))
                            temp_dm = np.zeros((2,2))
                        
                            temp_df[0][0] = Df[i*U_size] #S_prime[i*State_size]
                            temp_df[1][0] = Df[i*U_size+1] #S_prime[i*State_size+1]
        
                            #//copy M*S_prime for the matrix calculation of m.  These multiplications are performed using BLAS routines.  7page
                            #scipy.linalg.blas.dgemv(chn, m_m, n_m, one, M_dynamics[i*State_size*State_size], m_m, S_prime[i*State_size], p_m, zero, datam, p_m)
                            #print(i)
                            #print(len(S_prime[i*State_size]))
                            #print(M_dynamics[i*State_size*State_size])
                            #print(temp_m, temp_s)
                            #fblas.dgemv(one,temp_m, temp_s,zero,temp_d,overwrite_y = 1)#,m_m,p_m,n_m,p_m, 0,0) #, m_m) #  chn, m_m, n_m, , , m_m,, p_m, zero, datam, p_m)
                            #scipy.linalg.blas.dgemv(one, M_dynamics, S_prime ,zero,datam,m_m,p_m,n_m,p_m, chn, m_m)
                            temp_dm = np.dot(temp_df,temp_df.T)
                            
                            #print("hi", alpha )
                            A = np.reshape(temp_dm,(4)) + A
                            #print(A)
                            #datam = np.reshape(temp_dm,(2))
                            
        
                            #fblas.dsyr(uplo, &U_size, &alpha, &Df[i*U_size], &one, A, &U_size) #what the FUCK:(  can I find python blas library?? u should check 
                            #fblas.dsyr(alpha, Df[i*U_size], 0, one, U_size, A, 0)
                            
                            H_barrier[i*block_length+(2+U_size)*2+2] = kappa*(A[0]+1/(fx2[i]*fx2[i]))
                            H_barrier[i*block_length+(2+U_size)*2+3] = kappa*A[2] #//note these are both 2, because the result is stored in the upper triangular portion.
                            H_barrier[i*block_length+(2+U_size)*3+2] = kappa*A[2]
                            H_barrier[i*block_length+(2+U_size)*3+3] = kappa*A[3]
                        #print(H_barrier[i*block_length+(2+U_size)*2+2],H_barrier[i*block_length+(2+U_size)*2+3],H_barrier[i*block_length+(2+U_size)*3+2],H_barrier[i*block_length+(2+U_size)*3+3])
                    A=[]
                #print((H_barrier))
                if (indicator==0 or indicator ==2) and status:
                    for i in range(S_length-1):
                        G_barrier[(2+U_size)*i] = 0.0 #//zeroing the allocated memory
                        G_barrier[(2+U_size)*i+1] = 0.0
                        G_barrier[(2+U_size)*i+2]=(-1.0)*kappa*(Df[U_size*i]/fx[i]+1/fx2[i])
                        G_barrier[(2+U_size)*i+3]=(-1.0)*kappa*Df[U_size*i+1]/fx[i]
                #print(len(G_barrier))
                Df.clear()
            else:
                #print(fx2)
                for i in range(S_length-1):
                    # print((-1)*fx[i],(-1)*fx2[i])
                    F_barrier[i] = (-1)*kappa*(math.log((-1)*fx[i])+math.log((-1)*fx2[i]))
                    #print(F_barrier[i])
            
        # if cnt ==1:
        #     for i in range(S_length-1):
        #         print(F_barrier[i])
        # if(cnt ==1):
        #     for i in range(S_length-1):
        #         print("kaappa ",kappa,fx[i],fx2[i])
                    
    
    fx.clear()
    fx2.clear()
    if status:
        return valid, F_barrier, u 
    else:
        return -1

   