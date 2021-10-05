#import "MTSOS"
import math

#dynamics for a friction circle car, or 2 degree of freedom spacecraft model

def so_dynamics(S_middle,S_prime, S_length, State_size, U_size, variables, variables_length, R_dynamics, M_dynamics, C_dynamics, d_dynamics):
    m = variables[0]
    the = []
    #print('dy')
    
    block_length = U_size*State_size
    for i in range(S_length-1):
        theta = math.atan2(S_prime[State_size*i],S_prime[State_size*i+1]) # absolute angle
        the.append(theta)
        #print(theta)
        R_dynamics[i*block_length]=math.cos(theta)
        R_dynamics[i*block_length+1]=math.sin(theta)
        R_dynamics[i*block_length+2]=math.sin(theta) *(-1)
        R_dynamics[i*block_length+3]=math.cos(theta)

    block_length = U_size*State_size
    for i in range(S_length-1):
        M_dynamics[i*block_length]=m
        M_dynamics[i*block_length+1]=0.0
        M_dynamics[i*block_length+2]=0.0
        M_dynamics[i*block_length+3]=m
    
    for i in range((S_length-1)*State_size*State_size):
        C_dynamics[i] = 0.0
    
    for i in range((S_length-1)*State_size):
        d_dynamics[i] = 0.0
    
    # print("hihihih",the)
    return 1


