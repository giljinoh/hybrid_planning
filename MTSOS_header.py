#ifndef MTSOS_H_
#define MTSOS_H_

#include " parse.h"
#include <stdlib.h>


class algorithm_params:
    def __init__(self):
        self.kappa = 0
        self.alpha = 0
        self.beta = 0
        self.epsilon = 0
        self.MAX_ITER = 100

class problem_params:
    def __init__(self):
        self.S = []
        self.initial_velocity = 0
        self.State_size = 2
        self.U_size = 2

class algorithm_flags:
    def __init__(self):
        self.kappa = 0
        self.display = 0
        self.timer = 0

class optional_params:
    def __init__(self):
        self.variables = []
        self.variables_length = 0
        self.initial_b = 0
        self.initial_u = 0


def so_calcHbarrier(H_barrier, S_length,  U_size,  index_i,  index_j,  index_v):
    pass
def so_MakeHandG( b,  S_middleArray,  S_prime,  S_dprime,  u,  a,  S_length,  U_size,  State_size,  dtheta,  kappa,  flag_calculate, \
     G,  pH,  pindexH_i,  pindexH_j,  pindexH_v,  timers,  istimed,  variables,  variables_length):
     pass
def so_linesearch_infeasible( w,  v,  x,  dx,  b,  S_middle, S_prime, \
     S_dprime,  u,  a,   S_length,   U_size,   State_size,   dtheta,   kappa,   beta,\
           alpha,    G,  A21,    b_Ax,    variables,   variables_length):
           pass
def so_function_evaluation(   b,    S_middle,    S_prime,    S_dprime,    u,\
        a,   S_length,   U_size,   State_size,   dtheta,   kappa,    variables,   variables_length):
        pass
def so_linesearch_feasible(   w,    v,    x,    dx,    b,    S_middle, \
       S_prime,   S_dprime,    u,    a,   S_length,   U_size,   State_size,   dtheta,   kappa,    G,   alpha,   beta,   fx,   \
            variables,   variables_length):
            pass
            
def so_MakeA(   S,    S_middle,    S_prime,    S_dprime,   theta,   U_size, \
      S_lengt,   State_size,     pAc,     pA_i,     pA_j,     pA_v,     pb_Ax,   b_0, \
           variables,   variables_length):
           pass
def so_lu_solve(   Abig,   bbig,  S,  orderOkay):
    pass
def so_dynami (     S_middle,      S_prime,   S_length,   State_size, \
      U_size,     variables,   variables_length,    R_dynami ,    M_dynami , \
           C_dynami ,    d_dynami ):
           pass
def so_stuffing(   Abig_v,    Abig_i,    Abig_j,    indexH_v,    indexH_i, \
       indexH_j,   Hindex_length,    indexA21_v,    indexA21_i,    indexA21_j,   A21index_length, \
          Hc_m,   S_length,   State_size,   U_size):
          pass
def so_MTSOS(  p_params,   flags,   a_params,   o_params, \
        p_b,     p_u,     p_v,    p_fx,     p_timers,    piterations):
        pass

#endif
