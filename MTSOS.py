import dynamics
import barrier
import csparse
import math
#from scipy.linalg import _fblas as fblas
import numpy as np

cnt = 0
def so_MTSOS(p_params,  flags,  a_params,  o_params,  p_b,  p_u,  p_v,  p_fx,  p_timers,  p_iterations):
    global cnt
    #//the primary function that performs optimization on the provided vector S
    
    kappa, alpha, beta, mu, epsilon, sum, residual, residual_new, Lambda, time, b_0 = 0,0,0,0,0,0,0,0,0,0,0
    S_middle, b, a, x, u, v, S = [],[],[],[],[],[],[]
    S_prime, S_dprime, resid, data, Abig_v, bbig, dx, G, indexH_v, indexA21_v, b_Ax, timers, variables = [],[],[],[],[],[],[],[],[],[],[],[],[]
    dtheta = 0
    indexH_i, indexH_j, indexA21_i, indexA21_j = [],[],[],[]
    iterations, solved, feasible, display = 0,0,0,0
    S_length, State_size, index_length, U_size, Hindex_length, A21index_length, variables_length = 0,0,0,0,0,0,0
    i, j, status, totime, orderOkay = 0,0,0,0,0
    Abig_i, Abig_j = [],[]
    MAX_ITERATIONS, flags_kappa = 0,0
    RESIDUAL_THRESHHOLD, fx = 0,0
    cs = csparse.cs()
    csA21 = csparse.cs()
    Abig = csparse.cs()
    Abig_c = csparse.cs()
    Hc = csparse.cs()
    lu_S = csparse.css()
    
    status = 1 #;//flag that says everything is alright
   
    #//determine kappa, for the barrier function;
    
    if a_params.kappa <= 0:
        kappa = .1
    else:
        kappa = a_params.kappa

    #//determine the backtracking line search parameter alpha if set
    if a_params.alpha <= 0 or a_params.alpha >= .5:
        alpha = .01
    else:
        alpha = a_params.alpha
    
    #//determine the backtracking line searh parameter beta if set
    if a_params.beta <= 0 or a_params.beta >= 1:
        beta = .75
    else:
        beta = a_params.beta

    #//determine the espilon, accuracy term
    if a_params.epsilon <= 0:
        epsilon = .1
    else:
        epsilon = a_params.epsilon

    #//set the maximum number of iterations
    if a_params.MAX_ITER <= 0:
        MAX_ITERATIONS = 100
    else:
        MAX_ITERATIONS = a_params.MAX_ITER
    
    #//see if any optional variables have been set
    if o_params == None or o_params.variables == None:
        variables = None
        variables_length = 0
    else:
        variables = o_params.variables
        variables_length = o_params.variables_length
    
    
    mu = 10 #//the increase term for the barrier funciton
    solved = 0 #;//has a solution been found yet;
    feasible = 0 #;//is the current solution feasible
    
    fx = -1 #;//initialize fx to -1 to show that no feasible solution was found
    Abig_v = None #;//initialize Abig to NULL to show that space has not been allocated yet;
    bbig = None #;//iniitalize bbig to NULL to show the same has not been allocated
    iterations = 0 #;//you have not yet performed any iterations

    RESIDUAL_THRESHHOLD = 1e-6 #;//error threshhold required for the problem to be feasible.
    orderOkay = 0 #;//flag for detecting anamolies in factorization requiring refactorization
    lu_S = None #;//for storing factorization
    residual = math.inf #;//the residual has not yet been calculated
    
    U_size = p_params.U_size
    S_length = p_params.S_length
    State_size = p_params.State_size
    
    #//defaults to dynamic kappa (flags_kappa = 1);
    if flags.kappa != 0:
        flags_kappa = 1
    else:
        flags_kappa = 0

    #//defaults to no display (display = 1)
    if flags.display != 1:
        display = 0
    else:
        display = 1
        
        
    dtheta = 1.0/(S_length-1) #;//calcuate the length of the interval
    #//calculate S_middle;
    S = p_params.S
    S_middle = [0] * State_size*(S_length-1)  #calloc(State_size*(S_length-1),sizeof(double));
    S_prime = [0]  * State_size*(S_length-1) #calloc(State_size*S_length-1,sizeof(double));
    S_dprime = [0] * (S_length-1)*State_size #calloc((S_length-1)*State_size, sizeof(double));

    if S_middle == None or S_prime == None or S_dprime ==None:
        print("insufficient memory")
        status = 0
    else:
        for i in range(S_length-1):
            for j in range(State_size):
                S_middle[i*State_size+j]=(S[i*State_size+j]+S[(i+1)*State_size+j])/2 #;//S_middle = (S(:,2:end)+S(:,1:end-1))/2
                S_prime[i*State_size+j] = (S[(i+1)*State_size+j]-S[i*State_size+j])/dtheta #;//S_prime = (S(:,2:end)-S(:,1:end-1))/(dtheta)
                if S_length ==2:
                    S_dprime[0] = 0
                elif i == 0:
                    S_dprime[i*State_size+j] = (S[i*State_size+j]/2-S[(i+1)*State_size+j]+S[(i+2)*State_size+j]/2)/(dtheta*dtheta)
                elif i==S_length-2:
                    S_dprime[i*State_size+j] = (S[(i-1)*State_size+j]/2-S[i*State_size+j]+S[(i+1)*State_size+j]/2)/(dtheta*dtheta)
                elif i == 1 or i==S_length-3:
                    S_dprime[i*State_size+j] = (S[(i-1)*State_size+j]-S[i*State_size+j]-S[(i+1)*State_size+j]+S[(i+2)*State_size+j])/(2*dtheta*dtheta)
                else:
                    S_dprime[i*State_size+j] = ((-1.0)*S[(i-2)*State_size+j]*5/48+S[(i-1)*State_size+j]*13/16-S[i*State_size+j]*17/24-S[(i+1)*State_size+j]*17/24+S[(i+2)*State_size+j]*13/16-S[(i+3)*State_size+j]*5/48)/(dtheta*dtheta)

    
    #//initialize b to speed_constant/norm(S_prime)
    if p_b == None:
        b = [0] * S_length #calloc(S_length,sizeof(double));
    else:
        b  = p_b #realloc(*p_b, S_length*sizeof(double));
    a = [0] * (S_length-1)  #calloc(S_length-1,sizeof(double));
    if b==None or a == None:
        status = 0

    sum = 0

    for i in range(State_size):
        sum += (S[State_size+i]-S[i])*(S[State_size+i]-S[i])
    if p_params.initial_velocity < 0:
        print('negative velocity')
    
    b_0 = p_params.initial_velocity*p_params.initial_velocity*dtheta*dtheta/sum
    b[0] = b_0
    
    #//initialize the rest of the b vector, provided it was allocated

    if b != None:
        if o_params == None or o_params.initial_b == None:
            for i in range(1,S_length):
                b[i] = b_0/2+dtheta*dtheta
            
        else:
            for i in range(S_length-1):
                b[1+i] = o_params.initial_b[1]


    
    #//set all of the pointers being passed to MakeA to NULL so that they will get allocated in MakeA
    csA21 = None # ;//a csparse storing (compressed) of the dynamics constraints matrix.
    indexA21_i = None # ;//the i indices of the sparse A21 array
    indexA21_j = None # ;//the j indices of the sparse A21 array
    indexA21_v = None # ;//the values of the sparse A21 array;
    b_Ax = None # ;
    
    G = None # ;//space for holding the gradient.
    Hc = None # ;//space for holding the Hessian.
    indexH_i = None # ;//space for storing the i indices of the Hessian
    indexH_j = None # ;//space for storing the j indices of the Hessian
    indexH_v = None # ;//space for storing teh values of the Hessian
    
    # //set pointers to NULL so that you know what to free in the case of failure.
    Abig_c = None # ;
    Abig_i = None # ;
    Abig_j = None # ;
    dx = None # ;
    lu_S = None # ;
    
    if status:
        A21index_length, csA21, indexA21_i, indexA21_j, indexA21_v, b_Ax, b_0 = so_MakeA(S, S_middle, S_prime, S_dprime, dtheta, U_size, S_length, State_size, csA21, indexA21_i, indexA21_j, indexA21_v, b_Ax, b_0, variables, variables_length)
        
        if A21index_length == -1:
            status = 0
        
    #//make x and u should occur after the make A21 so that I can make sure they are the correct size;
    x = [0] * (csA21.n) #calloc(csA21->n,sizeof(double));
    resid = [0] * (csA21.m) #calloc(csA21->m,sizeof(double));
    if p_u == None:
        u = [0] * ((csA21.n)-2*(S_length-1)) # calloc(csA21->n-2*(S_length-1),sizeof(double));
        #print(u)
    else:
        u  = p_u #realloc(*p_u, csA21->n-2*(S_length-1)*sizeof(double));
    if p_v == None:
        v = [0] * (csA21.m) # calloc(csA21->m,sizeof(double));
    else:
        v = p_v #realloc(*p_v, csA21->m*sizeof(double));
    if x == None or u == None or v == None or resid == None:
        status = 0
    
    
    if status:
        #//check if an initial u has been provided, otherwise, initialize to zero
        if o_params == None or o_params.initial_u == None:
            for i in range(csA21.n-2*(S_length-1)):
                u[i] = 0.0
    
        else:
            for i in range(csA21.n-2*(S_length-1)):
                u[i] = o_params.initial_u[i]
        
        #//initialize the v vector to zero
        for i in range(csA21.m):
            v[i] = 0.0
        
        #//initialize the b terms in x generated earlier.
        for i in range(S_length-1):
            x[i*(U_size+2)] = b[i+1]
            x[i*(U_size+2)+1] = (b[i+1]-b[i])/(2*dtheta)
            a[i] = x[i*(U_size+2)+1]
            for j in range(U_size):
                x[i*(U_size+2)+2+j] = u[i*U_size+j]
    

    while not solved and iterations < MAX_ITERATIONS and status:
        print("cnt", iterations, status, solved)
       
        # if G != None:
        #     for i in range(len(G)):
        #         G[i] = round(G[i],6)
        # if b != None:
        #     for i in range(len(b)):
        #         b[i] = round(b[i],6)
        # if indexH_v != None:
        #     for i in range(len(indexH_v)):
        #         indexH_v[i] = round(indexH_v[i],6)
        # if u != None:
        #     for i in range(len(u)):
        #         u[i] = round(u[i],6)
        #print(b)
        iterations += 1
        cnt +=1
        #//calculate the Hessian and Gradient
        # print(" b ", b[1], end=" ")
        
        Hindex_length , G, Hc, indexH_i, indexH_j, indexH_v, u, b = so_MakeHandG(b, S_middle, S_prime, S_dprime, u, a, S_length, U_size, State_size, dtheta, kappa, 0, G, Hc, indexH_i, indexH_j, indexH_v,timers,totime,variables,variables_length)
        # if cnt ==3:
        #     print(b)
        #print(Hc.x)
        #print("k",u)
        # for i in range(100):
        #     print(indexH_v[i], end="")
        # for i in range((2+U_size)*(S_length-1)):
        #     print(-G[i],end=" ")
         
        if Hindex_length == -1:
            status = 0
        else:
            status = 1
        #print("U", u)
        # if cnt <10 :
        #     print("G",(G))
        #print(Hc.m)
           
        '''if(totime){//time for HandG
            diff = clock()-start;
            time = diff*1000.0/CLOCKS_PER_SEC;
            timers[2] += time;
            start = clock();
        }'''
        
        #//if the problem was infeasible last iteration, check if it feasible this iteration
        if not feasible and status:
            #//calculate A21*x-b_Ax;
            residual_new = 0.0
            for i in range(csA21.m):
                #print(b_Ax)
                resid[i] = (-1.0)*b_Ax[i]
            
            #if cnt == 1:
                #print(x)
            csparse.cs_gaxpy(csA21, x, resid)
            # if cnt ==1:
            #     print(resid[0])
            #     print(csA21.nz)
            #     print(csA21.m)
            #     print(csA21.n)
            #     print(csA21.nzmax)
            for i in range(csA21.m):
                #print(resid[i], RESIDUAL_THRESHHOLD)
                residual_new += resid[i]*resid[i]
            
            residual_new = math.sqrt(residual_new)
            
            
            if residual_new > residual:
                orderOkay = 0
            
            residual = residual_new
            
            if residual < RESIDUAL_THRESHHOLD:
                print("1")
                feasible = 1
                #//note that this is the first time function evaluation is called, so this can be thought of like initializaiton.
                fx = so_function_evaluation(b, S_middle, S_prime, S_dprime, u, a, S_length, U_size, State_size, dtheta,kappa, variables,variables_length)
                print("fx", fx)
                if fx == -1:
                    status = 0
                
        
            print("2")
        
        #//Now generate the large A matrix for the algorithm step.
        #//to do this, use the index arrays that are returned from make H and A12
        #//the arrays are the same size on every iteration, so you only need to allocate them once;
        if Abig_v == None and status:
            index_length = A21index_length*2+Hindex_length
            #print("len",index_length)
            Abig_i = [0] * index_length #  calloc(index_length,sizeof(int));
            Abig_j = [0] * index_length # calloc(index_length,sizeof(int));
            Abig_v = [0] * index_length #calloc(index_length,sizeof(double));
            bbig = [0] * (Hc.m+(S_length-1)*(State_size+1)) #calloc(Hc->m+(S_length-1)*(State_size+1),sizeof(double));//bbig;
            dx = [0] * ((S_length-1)*(2+U_size)) #calloc((S_length-1)*(2+U_size),sizeof(double));//dx from the solution;
            if(Abig_i == None or Abig_j == None or Abig_v == None or bbig == None or dx == None):
                status = 0
            
        
        #print("dx4 ", dx[1])
        #//stuff the matrix Abig.
        # for i in range(100):
        #     print(Abig_v[i], end = "")
        if status:
            Abig_v,  Abig_i,  Abig_j = so_stuffing(Abig_v, Abig_i, Abig_j, indexH_v, indexH_i, indexH_j, Hindex_length, indexA21_v, indexA21_i, indexA21_j, A21index_length, Hc.m, S_length, State_size, U_size)
            
            print("3")
            #//construct the large sparse array
            Abig.nzmax = index_length
            Abig.i = Abig_i
            Abig.p = Abig_j
            Abig.x = Abig_v
            Abig.nz = Abig.nzmax
            Abig.m = Hc.m+csA21.m
            Abig.n = Hc.n+csA21.m
            '''if(totime){
                start2 = clock();
            }'''
            #Abig_c = csparse.cs_triplet(Abig)
            #print("Abig", Abig.x)
            Abig_c = csparse.cs_compress(Abig)
            # print("Abig_c", Abig_c.x)
            '''if(totime){
                diff = clock()-start2;
                time = diff*1000.0/CLOCKS_PER_SEC;
                timers[18] += time;
            }'''
            if Abig_c == None:
                status = 0
            
        

        if status:
            #print("AAAA",Abig_c.nzmax)
            csparse.cs_dupl(Abig_c)
            #print("AAA",Abig_c.nzmax)
            csparse.cs_dropzeros(Abig_c)
            #print("AA",Abig_c.nzmax)
            
            #//finish creating b;
            #print(G)
            for i in range((2+U_size)*(S_length-1)):
                bbig[i] = G[i]*(-1.0) #;//-G potion
                # print(bbig[i],end=" ")
            # print(csA21.m)
            
            if not feasible: #//if not feasible the rest of this array should be zeros and will be set earlier.
                for i in range(csA21.m):
                    bbig[i+(2+U_size)*(S_length-1)]= resid[i]*(-1.0)
                    # print( resid[i]*(-1.0)," ",end=" ")
            else:
                #//since the solve overwrites bbig, we now need to set the rest of the vector to zero each time if it is zero;
                for i in range(csA21.m):
                    bbig[i+(2+U_size)*(S_length-1)]= 0.0
            
     
            # if cnt <5 :
            #     print("as",(bbig))
            print("4")
            # if cnt ==1 :
            #     print("Abig_c", Abig_c.x)
            # if cnt < 2:
            #     print("bbig2", bbig)
            orderOkay,Abig_c, bbig,lu_S  = so_lu_solve(Abig_c, bbig, lu_S, orderOkay)
            # if cnt < 10:
            #     print("Abig_c", Abig_c.x)
            #     print("bbig2", bbig)
            #print("lu", bbig)
            print("5")
            if orderOkay==0:
                status = 0
            
            
            '''if(totime){//time for solve
                diff = clock()-start;
                time = diff*1000.0/CLOCKS_PER_SEC;
                timers[5] += time;
                start = clock();
            }'''
        
        
        
        if status:
            #print("6")
            #//copy data into the dx array;
            for i in range((S_length-1)*(2+U_size)):
                dx[i] = bbig[i]
            
            #print(dx)
            #print("asas",dx)
            #print(feasible)
            if not feasible:  #-------------------------------------------------------------------------------------------------
            #if cnt == 1:
                
                print("7")
                #print("dx1",dx[1])
                status,v,x,dx,b,bbig = so_linesearch_infeasible(bbig, v, x, dx, b, S_middle, S_prime, S_dprime, u, a, S_length, U_size, State_size, dtheta, kappa, beta, alpha, G, csA21, b_Ax, variables, variables_length)
                #print("dx2",dx[1])
                #print("status", status)
                '''if(totime){//time for infeasible line search
                    diff = clock()-start;
                    time = diff*1000.0/CLOCKS_PER_SEC;
                    timers[6] += time;
                    start = clock();
                }'''
            else:
                print("12")
                #print("asd")
                #//calculate Lambda = dx'*H*dx;
                data = [0.0] * (S_length-1)*(2+U_size) #calloc((S_length-1)*(2+U_size), sizeof(double));
                if(data == None):
                    status = 0
                
                if status:
                    #//initialize data to be zero;
                    for i in range((S_length-1)*(2+U_size)):
                        data[i]=0
                    csparse.cs_gaxpy(Hc, dx, data) #y = A*x+y
                    #print(Hc.x)
                    Lambda = 0
                    print("dx3",dx[1])
                    for i in range((S_length-1)*(2+U_size)):
                        #print(round(dx[i],6), end ="")
                        
                        Lambda += data[i]*dx[i]
                    print("\n")
                    #print("Lambda/2",Lambda)    
                    data.clear()
                    
                
                if status:
                    #print("Lambda",Lambda,"epl",epsilon)
                    #//check Lambda to see if you're done and if kappa needs to be reduced.
                    print("Lambda/2",Lambda,"epsilon",epsilon , kappa)
                    # if Lambda != None:
                    #     round(Lambda,6)
                    if Lambda/2 < epsilon:
                        print("13")
                        if kappa*S_length > epsilon and (flags_kappa == 1):
                            kappa = kappa/mu
                            print("8")
                            #print("asd",b)
                            fx = so_function_evaluation(b, S_middle, S_prime, S_dprime, u, a, S_length, U_size, State_size, dtheta, kappa, variables, variables_length)
                            if fx == -1:
                                status = 0
                            
                        else:
                            print("9")
                            solved = 1
                           
                    else:
                        #//perform the feasible line search
                        print("10")
                       
                        fx,v, x, dx, b, bbig = so_linesearch_feasible(bbig, v, x, dx, b, S_middle, S_prime, S_dprime, u, a, S_length, U_size, State_size, dtheta, kappa, G, alpha, beta, fx, variables, variables_length)
                        # fx = so_linesearch_feasible(bbig, v, x, dx, b, S_middle, S_prime, S_dprime, u, a, S_length, U_size, State_size, dtheta, kappa, G, alpha, beta, fx, variables, variables_length)
                        print(fx)
                        if fx == -1:
                            status = 0
                    
                print("11")

        
        #//free the large A matrix you create
        if(Abig_c == None):
            #csparse.cs_spfree(Abig_c)   -----------------------------------------------------------
            Abig_c = None
        
        
        print("2cnt", iterations, status, solved)

    #//Set all of the values to be returned
    p_b = b
    p_u = u
    p_v = v
    p_fx = fx
    p_iterations = iterations
    
    if S_middle != None:
        S_middle.clear()
    if(S_prime!= None):
         S_prime.clear()
    if(S_dprime != None):
         S_dprime.clear()
    
    #//free all of the space allocated for csA21;
    if(indexA21_i != None):
        indexA21_i.clear()
    if(indexA21_j != None):
        indexA21_j.clear()
    if(indexA21_v != None):
        indexA21_v.clear()
    if(b_Ax != None):
        b_Ax.clear()
    if(csA21 != None):
        #csparse.cs_spfree(csA21)
        csA21.p = []
        csA21.i = []
        csA21.x = []
    
    if(a != None):
        a.clear()
    if(x !=  None):
        x.clear()
    #//destory arrays that are constant for HG
    if(G == None):
        G.clear()
    
    if(Hc != None):
        #csparse.cs_spfree(Hc)
        Hc.p = []
        Hc.i = []
        Hc.x = []
    if(indexH_i != None):
        indexH_i.clear()
    if(indexH_j != None):
        indexH_j.clear()
    if(indexH_v != None):
        indexH_v.clear()
    
    #//clear the elements created for the use in calculating the residual;
    if(resid != None):
        resid.clear()
    
    #//free space created to make Abig
    if(Abig_i !=None):
        Abig_i.clear()
    if(Abig_j != None):
        Abig_j.clear()
    if(Abig_v != None):
        Abig_v.clear()
    if(bbig != None):
        bbig.clear()
    if(dx != None):
        dx.clear()
    if(lu_S != None):
        lu_S = None
        #csparse.cs_sfree(lu_S)
    
    '''if(totime){//time for solve
            diff = clock()-start;
            time = diff*1000.0/CLOCKS_PER_SEC;
            timers[8] = time;
    }'''
    
    return status, p_fx, p_b


def so_calcHbarrier( H_barrier,  S_length,  U_size,  pindex_i,  pindex_j,  pindex_v):
    global cnt
    '''/*so_calcHBarrier, calculates the portion of the Hessian produced by the user defined barrier function.
 *BarrierArray: the three dimensional array of inputs produced by the barrier function
 *index_i: an array of integers to store the i indices
 *index_j: an array of integers to store the j indices
 *index_v: an array of doubles to store the values corresponding to those indices;
 *return: the length of the index arrays, or -1 if failure;
*/'''

           
    i, n, m, block_length, index_length = 0,0,0,0,0
    #status = 0
    index_v = []
    index_i, index_j = [],[]
    
    status = 1
    
    block_length = (2+U_size)*(2+U_size) #;//the size of each H block.
    index_length = block_length*(S_length-1) #;//the length of H_Barrier.
    
    #//note that to speed things up further, the i and j calculations need only be performed once, and could be stored in a different implementation.
    if pindex_i == None:
        index_i = [0] * index_length # calloc(index_length,sizeof(int));
    else:
        index_i = pindex_i # realloc(*pindex_i,index_length*sizeof(int));
    
    if pindex_j == None:
        index_j = [0] * index_length #calloc(index_length,sizeof(int));
    else:
        index_j = pindex_j # realloc(*pindex_j,index_length*sizeof(int));
    
    if pindex_v == None:
        index_v = [0.0] * index_length #calloc(index_length,sizeof(double));
    else:
        index_v = pindex_v #realloc(*pindex_v,index_length*sizeof(double));
    
    if index_i == None or index_j == None or index_v == None:
        status = 0

    
    
    pindex_i = index_i
    pindex_j = index_j
    pindex_v = index_v

    

    
    for i in range(S_length-1):
        #//copy the H from one time step into TempH so that it can be manipulated to account for the averaging of b across timesteps.
        for n in range(U_size+2):
            #//populate the indices of the block matrix.
            for m in range(U_size+2):
                index_i[i*block_length+n*(U_size+2)+m]=m+i*(U_size+2)
                index_j[i*block_length+n*(U_size+2)+m]=n+i*(U_size+2)

    
        
    #//copy the derivative data into index_v;
    for i in range((2+U_size)*(2+U_size)*(S_length-1)): # not sure
        index_v[i] = H_barrier[i]
    # if cnt ==1:
    #     print(pindex_v)
    #memcpy(index_v,H_barrier,index_length*sizeof(double));
    if(status):
        return index_length,pindex_i,  pindex_j,  pindex_v
    else:
        return -1

#staic time = 0
def so_MakeHandG( b,  S_middle,  S_prime,  S_dprime,  u,  a,  S_length,  U_size,  State_size,  dtheta,  kappa,  flag_calculate,  G,  pH,  pindexH_i,  pindexH_j,  pindexH_v,  timers,  istimed,  variables,  variables_length):
    global cnt
    '''/*b: an array of the b values
     *S_middle: the S values evaluated at their midpoints
     *u: an array of the u values as determined at the midpoints of the S values
     *dtheta: the distance along the path between steps
     *kappa: the weighting term for the barrier
     *barrier: the function handle used to call the barrier function
     *flag_calculate: a flag showing which terms (H and G) need to be calculated: 0 for both 1 for H, 2 for G
     *G: a pointer to an array for storing the gradient
     *H: a pointer to a csparse structure in which to store the hessian
     *indexH_i: a pointer to an array of ints to store the i index to make the KKT system with
     *indexH_j: a pointer to an array of ints store the j index to make the KKT system with
     *indexH_v: a pointer to an array of doubles to store the values corresponding with the indices in indexH_i and indexH_j for the creation of the KKT system
     *return:  the length of the indexH arrays.*/'''
    time = 0
    # time+=1
    doG, doH, status, statusB, statusF, length_Hb = 0,0,0,0,0,0
    i = 0 #//index for loops.
    index_length = 0
    indexH_i, indexH_j, indexHb_i, indexHb_j =[],[],[],[]
    data, bsqrt, bsqrt_mid, bsqrt_mid2, bsqrt3, bsqrt_mid3, indexH_v, indexHb_v, H_barrier, G_barrier, F_barrier = [],[],[],[],[],[],[],[],[],[],[]
    H = csparse.cs()
    Hc = csparse.cs() #;//Hc is compressed form H.
    #clock_t start, diff;
    '''if(istimed){
        start = clock();
        if(timers == NULL)
            istimed = 0;
    }'''
    #print("G",G)
    status = 1 #;//checks memory status
    statusB = 1
    statusF = 0
    index_length = 3*S_length-5+(U_size+2)*(U_size+2)*(S_length-1)
    H_barrier = None
    G_barrier = None
    F_barrier = None
    indexHb_i = None
    indexHb_j = None
    indexHb_v = None
    #//set to NULL so we know if they need to be freed in the case of memory failure
    bsqrt = None
    bsqrt_mid = None
    bsqrt_mid2 = None
    
    #//check to make sure that b>0 if not, then terminate
    for i in range(S_length):
        if statusB:
            if(b[i]<0):
                statusB = 0
    
    
    if(statusB):
        doG = ((flag_calculate == 0) or (flag_calculate == 2))
        doH = ((flag_calculate == 0) or (flag_calculate == 1))
    else:
        doG = 0
        doH = 0
    
    

    '''if(istimed){
        diff = clock()-start;
        time = diff*1000.0/CLOCKS_PER_SEC;
        timers[9] += time;//HG initialization;
        start = clock();
    }'''
    
    if(not doG):
        if(G == None):
            G = [0] #calloc(1, sizeof(double));
            if(G == None):
                status = 0
            
        if(status):
            G[0] = math.inf
        '''if(istimed){
            diff = clock()-start;
            time = diff*1000.0/CLOCKS_PER_SEC;
            timers[10] += time;//G infeasible;
            start = clock();
        }'''
    
    
    if(statusB):
        bsqrt = [None] * S_length #malloc(S_length*sizeof(double))
        bsqrt_mid = [None] * (S_length-1) #malloc((S_length-1)*sizeof(double))
        bsqrt_mid2 = [None] * (S_length-1) #malloc((S_length-1)*sizeof(double))
        #print(bsqrt == None or bsqrt_mid == None or bsqrt_mid2 == None)
        if(bsqrt == None or bsqrt_mid == None or bsqrt_mid2 == None):
            #print('adad')
            status = 0
        else:
            #//calculates some values used by everyone.
            for i in range(S_length): 
                bsqrt[i] = math.sqrt(b[i])
                # print(bsqrt[i] ,end="")
                if(i!=0):
                    bsqrt_mid[i-1] = bsqrt[i-1]+bsqrt[i]
                    bsqrt_mid2[i-1] = 1/(bsqrt_mid[i-1]*bsqrt_mid[i-1])
            #print("\n")    
            #print(bsqrt_mid)
        
        '''if(istimed){
            diff = clock()-start;
            time = diff*1000.0/CLOCKS_PER_SEC;
            timers[11] += time;//bqrt generation;
            start = clock();
        }'''
    #print(G[0])
    # for i in range(100):
    #     print(bsqrt[i])    
    #print("\n")
    
    #//calculates the terms in the Hessian and Gradient dependent on problem formulation, not on the barrier.  This may need to be decremented by 1.
    if(doH and status):
        bsqrt3 = [None] * (S_length-1) #malloc((S_length-1)*sizeof(double));
        bsqrt_mid3 = [None] * (S_length-1) #malloc((S_length-1)*sizeof(double));
        #//3*S_length-5 is for this term, the remainder of the space allocated is for the ouput from CalcH.
        if(pindexH_i==None):
            indexH_i = [None] * index_length # malloc(index_length*sizeof(int));
        else:
            indexH_i = pindexH_i # realloc(*pindexH_i, index_length*sizeof(int));//these reallocs may not be necessary, since they should be the same size.
        
        if(pindexH_j==None):
            indexH_j = [None] * index_length #malloc(index_length*sizeof(int));
        else:
            indexH_j = pindexH_j # realloc(*pindexH_j, index_length*sizeof(int));
        
        if(pindexH_v==None):
            indexH_v = [None] * index_length# malloc(index_length*sizeof(double));
        else:
            indexH_v = pindexH_v #realloc(*pindexH_v, index_length*sizeof(double));
        
        if(bsqrt3 == None or bsqrt_mid3 == None or indexH_i == None or indexH_j == None or indexH_v == None):
            
            status = 0
        else:
            #//generate the Hessian necessary for the minimum time portion of the problem.
            
            #print(dtheta)
            for i in range(S_length-1):
                #//generate some terms necessary for the calculation of the Hessian for the minimum time component.
                bsqrt3[i] = 1.0/(bsqrt[i+1]*bsqrt[i+1]*bsqrt[i+1])
                bsqrt_mid3[i] = 1.0/(bsqrt_mid[i]*bsqrt_mid[i]*bsqrt_mid[i])
                #//generate the diagonal elements of the Hessian.
                indexH_v[i] = dtheta*(bsqrt3[i]*bsqrt_mid2[i]/2+bsqrt_mid3[i]/b[i+1])+kappa/(b[i+1]*b[i+1])
                #print(bsqrt[i],  end="")
                indexH_i[i] = (2+U_size)*i
                indexH_j[i] = (2+U_size)*i
                #//generate the offdiagonal elements of the Hessian, that arise because we are looking at the midpoint between bs.
                if(i>0): #{//for all except the end term, so all elements are i-1, so that the b terms can be calculated in the same loop.
                    indexH_v[i-1]+= dtheta*(bsqrt3[i-1]*bsqrt_mid2[i-1]/2+1/b[i]*bsqrt_mid3[i-1])
                    indexH_v[i+S_length-2] = dtheta*(bsqrt_mid3[i]/(bsqrt[i]*bsqrt[i+1]))
                    indexH_i[i+S_length-2] = (2+U_size)*(i-1)
                    indexH_j[i+S_length-2] = (2+U_size)*(i)
                    indexH_v[i+2*S_length-4] = dtheta*(bsqrt_mid3[i]/(bsqrt[i]*bsqrt[i+1]))
                    indexH_i[i+2*S_length-4] = (2+U_size)*(i)
                    indexH_j[i+2*S_length-4] = (2+U_size)*(i-1)
                    
        
        #print(("indexH_v",indexH_v[0]))
        if(bsqrt3 != None):
            bsqrt3.clear()
        
        if(bsqrt_mid3 != None):
            bsqrt_mid3.clear()
        '''if(istimed):
            diff = clock()-start;
            time = diff*1000.0/CLOCKS_PER_SEC;
            timers[12] += time;//H problem formulation;
            start = clock();
        }'''
    
    
    
    #//perform setup necessary to call the barrier function
    if(statusB and status): #{//if the B status already failed then you aren't going to be doing anything.
        if(flag_calculate<2):
            H_barrier = [None] * (2+U_size)*(2+U_size)*(S_length-1) #malloc((2+U_size)*(2+U_size)*(S_length-1)*sizeof(double));
            if(H_barrier == None):
                status = 0
            
        
        if(flag_calculate == 0 or flag_calculate == 2):
            G_barrier = [None] * (2+U_size)*(S_length-1) #malloc((2+U_size)*(S_length-1)*sizeof(double));
            if(G_barrier == None):
                status = 0
            
        
        if(status):
            # if cnt ==1:
            #     print(u)
            # if cnt ==1:
            #     print(H_barrier)
            #print(u)
            statusF, F_barrier, u = barrier.so_barrier(S_middle, S_prime, S_dprime, b[1], a, u, S_length, U_size, State_size, flag_calculate, kappa, variables, variables_length, H_barrier, G_barrier, F_barrier, cnt)
            
            # if cnt ==1:
            #     print((G_barrier))
            if(statusF == -1):
                status = 0
                statusF = 0
            
        
        '''if(istimed){
            diff = clock()-start;
            time = diff*1000.0/CLOCKS_PER_SEC;
            timers[13] += time;//barrier function;
            start = clock();
        }'''
    
    
    
    if(doG and status):
        if(statusF == 0):
            if(G == None):
                G = [0] #calloc(1, sizeof(double));
                if(G == None):
                    status = 0
                
            
            G[0] = math.inf
        else:
            #//allocate space for G if it has not yet been allocated.
            if(G == None):
                G = [0] * (2+U_size)*(S_length-1) # calloc((2+U_size)*(S_length-1), sizeof(double));
                if(G == None):
                    status = 0
                data = G
            else:
                G = G #realloc(*G, (2+U_size)*(S_length-1)*sizeof(double));
                if(G == None):
                    status = 0
                data = G
            
            if(status):
                #memcpy(data, G_barrier, (2+U_size)*(S_length-1)*sizeof(double));
                for i in range((2+U_size)*(S_length-1)):
                    data[i] = G_barrier[i]
                    # print(G_barrier[i], end = " ")
                for i in range(S_length-1):
                    #//populate the G1 terms which correspond to the minimum time calculation.
                    data[i*(2+U_size)] += dtheta*((-1)*bsqrt_mid2[i]/bsqrt[i+1])-kappa/b[i+1]
                    # print(  b[i+1], end=" ")    
                    # print( data[i*(2+U_size)], end=" ")  
                    if(i!=S_length-2):
                        data[i*(2+U_size)] += (-1)*dtheta*bsqrt_mid2[i+1]/bsqrt[i+1]
                        # print( (-1)*dtheta*bsqrt_mid2[i+1]/bsqrt[i+1], end=" ")
                        


            # for i in range(S_length-1):
                        
            
        
        '''if(istimed){
            diff = clock()-start;
            time = diff*1000.0/CLOCKS_PER_SEC;
            timers[14] += time;//G formulation;
            start = clock();
        }'''
    # if cnt == 1:
    #    print(G) 
    
    
    if(doH and status):
        #print(indexHb_v)
        length_Hb , indexHb_i,indexHb_j,indexHb_v = so_calcHbarrier(H_barrier, S_length, U_size, indexHb_i,indexHb_j,indexHb_v)
        #print(("indexH_v2",indexH_v[0]))
        # if cnt ==1:
        #     print(indexHb_v)
        if(length_Hb == -1):
            status = 0
        
        '''if(istimed){
            diff = clock()-start;
            time = diff*1000.0/CLOCKS_PER_SEC;
            timers[15] += time;//calcH;
            start = clock();
        }'''
       
        if(status):
            #memcpy(&indexH_v[3*S_length-5], indexHb_v, length_Hb*sizeof(double));
            #memcpy(&indexH_i[3*S_length-5], indexHb_i, length_Hb*sizeof(int));
            #memcpy(&indexH_j[3*S_length-5], indexHb_j, length_Hb*sizeof(int));

            #indexH_v[3*S_length-5] = indexHb_v
            #indexH_i[3*S_length-5] = indexHb_i
            #indexH_j[3*S_length-5] = indexHb_j

            indexH_v[3*S_length-5:3*S_length-5+length_Hb] = indexHb_v
            indexH_i[3*S_length-5:3*S_length-5+length_Hb] = indexHb_i
            indexH_j[3*S_length-5:3*S_length-5+length_Hb] = indexHb_j

            #print(indexH_v)

            
            H.nzmax = 3*S_length-5+length_Hb
            H.i = indexH_i
            H.p = indexH_j
            H.x = indexH_v
            H.nz = H.nzmax
            H.m = (S_length-1)*(2+U_size)
            H.n = (S_length-1)*(2+U_size)
            
            Hc = csparse.cs_compress(H)
            #Hc = csparse.CS_TRIPLET(H)
            if(Hc == None):
                status = 0
           
            csparse.cs_dupl(Hc)
            csparse.cs_dropzeros(Hc)
            #//Now I just need away to return this matrix.
            if(pH != None):
                pH = None
                #csparse.cs_spfree(pH)
                # pH.p = []
                # pH.i = []
                # pH.x = []
            
            pH = Hc
            
            #//make sure that all of the index arrays are returned
            pindexH_v = indexH_v
            pindexH_i = indexH_i
            pindexH_j = indexH_j
        
        if(indexHb_i != None):
            indexHb_i.clear()
        if(indexHb_j != None):
            indexHb_j.clear()
        if(indexHb_v != None):
            indexHb_v.clear()
        '''if(istimed){
            diff = clock()-start;
            time = diff*1000.0/CLOCKS_PER_SEC;
            timers[16] += time;//big H generation;
            start = clock();
        }'''
    
    
    if(statusB):
        if(H_barrier!=None):
            H_barrier.clear()
            
        if(G_barrier!=None):
            G_barrier.clear()
           
        if(F_barrier!=None):
            F_barrier.clear()
            
        if(bsqrt != None):
            bsqrt.clear()
            
        if(bsqrt_mid != None):
            bsqrt_mid.clear()
            
        if(bsqrt_mid2 != None):
            bsqrt_mid2.clear()
            
    
    
    
    
    '''if(istimed){
        diff = clock()-start;
        time = diff*1000.0/CLOCKS_PER_SEC;
        timers[17] += time;//freeing;
    }'''
    if(status):
        return index_length, G,  pH,  pindexH_i,  pindexH_j,  pindexH_v, u,b
    else:
        return -1


def so_linesearch_infeasible( w,  v,  x,  dx,  b,  S_middle,  S_prime,  S_dprime, u,  a,  S_length,  U_size,  State_size,  dtheta,  kappa,  beta,  alpha,  G,  A21,  b_Ax,  variables,  variables_length):
    #//inputs are w [0],v [1],x [2],dx [3], S_middle [4], dtheta [5], kappa [6], beta [7], G [8], A_21 [9], b_Ax[10], barrier [11], alpha[12];
    #//plhs[0] = X, plhs[1] = V, plhs[2] = b, because it is needed later;
    global cnt
    #print("aaa",A21.m)
    X, V, term1, term2, Gnew = [],[],[],[],[]
    t, LHS, RHS = 0,0,0
    MAX_ITERATIONS, iterations = 0,0
    i, j, status, doloop, calcRHS = 0,0,0,0,0
    indexH_i,indexH_j = [],[]
    indexH_v = []
    A21T_c = csparse.cs()
    Htrash = csparse.cs()
    
    t = 1.0
    doloop = 1
    calcRHS = 1
    Gnew = None
    Htrash = None
    indexH_i = None
    indexH_j = None
    indexH_v = None
    A21T_c = None
    term1 = None
    term2 = None
    MAX_ITERATIONS = 20
    iterations = 0
    status = 1
    
    X = [0] * A21.n #calloc(A21->n,sizeof(double));
    V = [0] * A21.m #calloc(A21->m,sizeof(double));
    if(X == None or V == None):
        status = 0
    
    
    if(status):
        while(True):
            #print("hihih")
            #//compute X = x+t*dx;
            for i in range(A21.n):
                X[i] = x[i]+t*dx[i]
                # print("x dx",x[i], dx[i])
            
            
            #//compute b and u (or just copy from the computation);
            for i in range(S_length-1):
                b[i+1] = X[i*(U_size+2)]
                a[i] = X[i*(U_size+2)+1]
                for j in range(U_size):
                    u[i*U_size+j]=X[i*(U_size+2)+2+j]
                # if cnt ==1:
                #     print(X)
            
            
            #//calculate V
            #print(w)
            for i in range(A21.m):
                V[i] = v[i]*(1-t)+t*w[(S_length-1)*(2+U_size)+i] 
                # V[i] = v[i]*(1-t)+t*w[800+i]
                # print("v dx",v[i], w[800+i])
            
            status,  Gnew, Htrash, indexH_i, indexH_j, indexH_v, u, b = so_MakeHandG(b, S_middle, S_prime, S_dprime, u, a, S_length, U_size, State_size, dtheta, kappa, 2, Gnew, Htrash, indexH_i, indexH_j, indexH_v, None, 0, variables, variables_length) #;//the last three are H_i, H_j, H_v
            #print("statue1",status)
            if(status == -1):
                status = 0
            else:
                status = 1
            
            if(status and not(Gnew[0]==math.inf)):
               # //perform right hand calculation, but only on the first time through;
                if(calcRHS):
                    RHS = 0
                    #//calculate A21*x-b_Ax;
                    term1 = [0.0] * A21.m #calloc(A21->m, sizeof(double));
                    if(term1 == None):
                        status = 0
                    
                    if(status):
                        for i in range(A21.m):
                            term1[i] = (-1)*b_Ax[i]
                        csparse.cs_gaxpy(A21, x, term1)
                        for i in range(A21.m):
                            RHS += term1[i]*term1[i]
                        
                        #//calculate G+A21'*v
                        A21T_c = csparse.cs_transpose(A21, 1)
                        if(A21T_c == None):
                            status = 0
                        
                        term2 = [0.0] * A21.n #calloc(A21->n, sizeof(double));
                        if(term2 == None):
                            status = 0
                        
                    #print("statue0",status)
                    if(status):
                        #memcpy(term2, G, A21->n*sizeof(double));
                        for i in range(A21.n):
                            term2[i] = G[i]
                        csparse.cs_gaxpy(A21T_c, v, term2)
                        for i in range(A21.n):
                            RHS += term2[i]*term2[i]
                        calcRHS = 0
                    
                
                LHS = 0
                #print(Gnew[0])
                if(status and not(Gnew[0]==math.inf)):
                    #//Calculate A21*X-b_Ax;
                    for i in range(A21.m):
                        term1[i] = (-1)*b_Ax[i]
                    csparse.cs_gaxpy(A21, X, term1)
                    for i in range(A21.m):
                        LHS += term1[i]*term1[i]
                    #print("LHS",LHS,"RHS",RHS)
                    #//Gnew+A21'*V
                    # memcpy(term2, Gnew, A21->n*sizeof(double));
                    for i in range(A21.n):
                        term2[i] = Gnew[i]
                    csparse.cs_gaxpy(A21T_c, V, term2)
                    for i in range(A21.n):
                        LHS+= term2[i]*term2[i]
                    
                    #//perform left hand calculation.
                    #//mexPrintf("%e<%e\n",LHS,(1-alpha*t)*(1-alpha*t)*RHS);
                    #t = 1.0
                    # print("alpha",alpha,"t",t)
                    #print("LHS",LHS,"RHS",(1-alpha*t)*(1-alpha*t)*RHS)
                    if(LHS<(1-alpha*t)*(1-alpha*t)*RHS):
                        # print("LHS",LHS,"RHS",(1-alpha*t)*(1-alpha*t)*RHS)
                        doloop = 0
                
            #print("statue2",status)
            #print("iter", iterations, MAX_ITERATIONS, doloop)
            #print("statue3",status)
            #print(doloop, iterations)
            if(doloop and iterations > MAX_ITERATIONS):
                doloop = 0
                status = 0
            
            iterations += 1
            # print(" d ",iterations)
            t *= beta #; //initialize t
            if (doloop and status): 
                continue
            break
    
    #print("statue3",status)
    #//mexPrintf("Gnew[0] is %f\n",Gnew[0]);
    #//copy X and V into x and v before freeing;
    if(status):
        #memcpy(x, X, A21->n*sizeof(double));
        #memcpy(v, V, A21->m*sizeof(double));
        for i in range(A21.n):
            x[i] = X[i]
        for i in range(A21.m):
            v[i] = V[i]
        #print(x,v)
    
    if(X != None):
        X.clear()
        
    if(V != None):
        V.clear()
        
    if(Gnew != None):
        Gnew.clear()
        
    if(A21T_c != None):
        #csparse.cs_spfree(A21T_c)
        A21T_c = None
    if(term1 != None):
        term1.clear()
        
    if(term2 != None):
        term2.clear()
        
    if(indexH_i != None):
        indexH_i.clear()
        
    if(indexH_j != None):
        indexH_j.clear()
        
    if(indexH_v != None):
        indexH_v.clear()
        
    if(Htrash != None):
        #csparse.cs_spfree(Htrash)
        Htrash = None
   
    return status,  v,  x, dx, b, w


def so_function_evaluation( b,  S_middle,  S_prime,  S_dprime,  u,  a,  S_length,  U_size,  State_size,  dtheta,  kappa,  variables,  variables_length):
    global cnt
    i, b_okay, status, valid = 0,0,0,0
    bsqrt,bsqrt_mid, F = [],[],[]
    
    fx = 0.0
    b_okay = 1
    bsqrt = [0.0] * S_length #calloc(S_length,sizeof(double));
    bsqrt_mid = [0.0] * (S_length-1) #calloc(S_length-1,sizeof(double));
    status = 1
    if(bsqrt == None or bsqrt_mid == None):
        status = 0
    
    
    for i in range(S_length):
        if b_okay:
            if(b[i]<0): #//if b[i] < zero trajectory is infeasible and value should be inf.
                b_okay = 0
            else: #{//calculate bsqrt and bsqrtmid for use in the funciton evaluation.  (in particular the b>0 barrier)
                bsqrt[i] = math.sqrt(b[i])
                if(i!=0):
                    bsqrt_mid[i-1] = bsqrt[i-1]+bsqrt[i]
        
    
    #print("fx 1", fx)
    if(b_okay and status):
        #//call the barrier function
        #//for user easer, the memory is allocated here so the user doesn't need to worry about memory allocation,
        #//However, this memory really only needs to be allocated once, so perhaps move up to an even higher level.
        F = [0.0] * (S_length-1) #malloc((S_length-1)*sizeof(double));
        #print("fx2", fx)
        valid,F, u = barrier.so_barrier(S_middle, S_prime, S_dprime, b[1], a, u, S_length, U_size, State_size, 3, kappa, variables, variables_length, None, None, F, cnt)
        #print("fx3", fx)
        #print("F",F)
        if(valid == -1):
            status = 0
        else:
            status = 1

        if(valid and status):
            for i in range(S_length-1):
                fx += 2.0*dtheta/bsqrt_mid[i]+F[i]-kappa*(math.log(b[i+1]))
            
        else:
            b_okay = 0
        #print("fx4", fx)
        #//free the results of the barrier function
        if(F!=None):
            F.clear()
    
    #//free the sqrt arrays that were just created.
    if(bsqrt != None):
        bsqrt.clear()
        
    if(bsqrt_mid != None):
        bsqrt_mid.clear()
    
    if(status):
        if(b_okay):
            return(fx)
        else:
            return(math.inf)
    else:
        return -1


def so_linesearch_feasible( w,  v,  x,  dx,  b,  S_middle,  S_prime,  S_dprime,  u,  a,  S_length,  U_size,  State_size,  dtheta,  kappa,  G,  alpha,  beta,  fx,  variables,  variables_length):
    t, RHS, fX = 0.0,0.0,0.0
    X = []
    status, i, j, iterations, ITERATIONS_MAX, doloop, calcRHS = 0,0,0,0,0,0,0
    
    status = 1
    t = 1.0
    iterations = 0
    ITERATIONS_MAX = 20
    doloop = 1
    calcRHS = 1
    
    X = [0] * ((S_length-1)*(2+U_size)) #calloc((S_length-1)*(2+U_size),sizeof(double));
    if(X == None):
        status = 0
    
    
    if(status):
        while(True):
            #//compute X = x+t*dx;
            for i in range((S_length-1)*(2+U_size)):
                X[i] = x[i]+t*dx[i]
                # if cnt < 5:
                #     print( "ihi", x[i],dx[i], end=" ")
            
            #//compute b, u (or just copy from the computation);
            for i in range(S_length-1):
                b[i+1] = X[i*(U_size+2)]
                # print( b[i+1], end=" ")
                a[i] = X[i*(U_size+2)+1]
                for j in range(U_size):
                    u[i*U_size+j]=X[i*(U_size+2)+2+j]
                
            
            
            fX = so_function_evaluation(b, S_middle, S_prime, S_dprime, u, a, S_length, U_size, State_size, dtheta, kappa, variables, variables_length)
            if(fX == -1):
                status = 0
            
            
            if(calcRHS):
                RHS = 0
                #//mexPrintf("G[0]=%f,dx[0]=%f\n",data[0],data2[0]);
                for i in range((S_length-1)*(U_size+2)):
                    RHS += G[i]*dx[i]
                
                #//mexPrintf("\n");
                RHS *= alpha
                calcRHS = 0
            
            
            #//mexPrintf("%e<%e\n",fX,fx+t*RHS);
           # print(fX)
            #print("fx", fx, "t", t,"RHS",RHS)
            if(fX != math.inf and fX < fx+t*RHS):
                doloop = 0
            else:
                iterations += 1
                t *= beta
            
            #print("iterrations",iterations)
            if(iterations > ITERATIONS_MAX):
                doloop = 0
                status = 0
            
            #print("doloop",doloop,"status",status)
            if(doloop and status): 
                continue
            break
    
    # print(" iter ", iterations)
    #//update X
    #memcpy(x,X,(S_length-1)*(U_size+2)*sizeof(double))
    for i in range((S_length-1)*(U_size+2)):
        x[i] = X[i]
    if(X != None):
        X.clear()
        #free(X);
    
    if(status):
        #//update V
        #print("w",len(w))
        for i in range((S_length-1)*(State_size+1)):
            #print("w",w[800+i])
            v[i] = v[i]*(1-t)+t*w[(S_length-1)*(2+U_size)+i]
            # print("v",v[i],end = " ")
        return fX,v,x, dx, b, w
        # return fX
    else:
        return -1


def so_MakeA( S,  S_middle,  S_prime,  S_dprime,  dtheta,  U_size,  S_length,  State_size,  pAc,  pA_i,  pA_j,  pA_v,  pb_Ax,  b_0,  variables,  variables_length):
    #//generates the matrices necessary to represent the dynamics
    #//if things fail, return -;
    #print(S)
    #print(len(S_prime))
    A_v, b_Ax = [],[]
    datac1, datac2, datam = [],[],[]
    M_dynamics, R_dynamics, C_dynamics, d_dynamics = [],[],[],[]
    A_i, A_j, indexR_i, indexR_j = [],[],[],[]
    block_length, index_length = 0,0 #;//size parameters
    i, j, status = 0,0,0 #;//indices for loops
    A = csparse.cs()
    m_m,n_m,p_m = 0,0,0 #;//variables for the use of mtimes;
    chn = 0 #"N" #;//our matrices for our matrix multiplies are in column order, so no transpose necessary.
    one = 1.0
    zero = 0.0 #;//values for use in the matrix muliply one*A*B+zero*C
    
    #//a flag to make sure everything has executed properly
    status = 1
    
    m_m = State_size
    n_m = State_size
    p_m = 1
    
    #//generate the first and second derivative of S with respect to theta necessary for the algorithm.
    #//S_prime = calloc((S_length-1)*State_size, sizeof(double));
    R_dynamics = [None] * (S_length-1)*U_size*State_size # malloc((S_length-1)*U_size*State_size*sizeof(double));
    M_dynamics = [None] * (S_length-1)*State_size*State_size #malloc((S_length-1)*State_size*State_size*sizeof(double));
    C_dynamics = [None] * (S_length-1)*State_size*State_size #malloc((S_length-1)*State_size*State_size*sizeof(double));
    d_dynamics = [None] * (S_length-1)*State_size #malloc((S_length-1)*State_size*sizeof(double));
    indexR_i = [0] * U_size*State_size #calloc(U_size*State_size,sizeof(int));
    indexR_j = [0] * U_size*State_size #calloc(U_size*State_size,sizeof(int));
    datam  = [0] * (m_m*n_m) #calloc(m_m*n_m,sizeof(double));
    datac1 = [0] * (m_m*n_m) #calloc(m_m*n_m,sizeof(double));
    datac2 = [0] * (m_m*n_m) #calloc(m_m*n_m,sizeof(double));
    if(R_dynamics == None or M_dynamics == None or C_dynamics == None or d_dynamics == None or indexR_i == None or indexR_j == None or datam == None or datac1 == None or datac2 == None):
        status = 0
    
    
    #//allocate space for the A matrix;
    block_length = (State_size*3+State_size*U_size+3) #;//4 for a,c*2;R;b'=a;
    index_length = block_length*(S_length-1)-State_size
    if(pA_i == None):
        A_i = [0] * index_length #calloc(index_length, sizeof(int));//this is written starting from rest...and thus b(0) is 1.
    else:
        A_i = pA_i #realloc(*pA_i, index_length*sizeof(int));
    
    if(pA_j == None):
        A_j = [0] * index_length #calloc(index_length, sizeof(int));
    else:
        A_j = pA_j #realloc(*pA_j,index_length*sizeof(int));
    
    if(pA_v == None):
        A_v = [0] * index_length #calloc(index_length, sizeof(double));
    else:
        A_v = pA_v #realloc(*pA_v,index_length*sizeof(double));
    
    if(pb_Ax == None):
        b_Ax = [0] * (S_length-1)*(State_size+1)#  calloc((S_length-1)*(State_size+1),sizeof(double));
    else:
        b_Ax = pb_Ax #realloc(*pb_Ax,(S_length-1)*(State_size+1)*sizeof(double));
    
    if(A_i == None or A_j == None or A_v == None or b_Ax == None):
        status = 0
    
    
    
    
    status = dynamics.so_dynamics(S_middle, S_prime, S_length, State_size, U_size, variables, variables_length, R_dynamics, M_dynamics, C_dynamics, d_dynamics)
    #print(R_dynamics)
    if(status):
        for i in range(U_size):
            for j in range(State_size):
                indexR_i[i*State_size+j] = j
                indexR_j[i*State_size+j] = i
    
    if(status):
        for i in range(S_length-1):
            temp_m = np.zeros((2,2))
            temp_s = np.zeros((2,1))
            temp_sd = np.zeros((2,1))
            temp_dm = np.zeros((2,2))
            temp_dc1 = np.zeros((2,2))
            temp_dc2 = np.zeros((2,2))
            temp_c = np.zeros((2,2))
            temp_m[0][0] = M_dynamics[i*State_size*State_size]
            temp_m[0][1] = M_dynamics[i*State_size*State_size+1]
            temp_m[1][0] = M_dynamics[i*State_size*State_size+2]
            temp_m[1][1] = M_dynamics[i*State_size*State_size+3]
            temp_c[0][0] = C_dynamics[i*State_size*State_size]
            temp_c[0][1] = C_dynamics[i*State_size*State_size+1]
            temp_c[1][0] = C_dynamics[i*State_size*State_size+2]
            temp_c[1][1] = C_dynamics[i*State_size*State_size+3]
            temp_s[0][0] = S_prime[i*State_size]
            temp_s[1][0] = S_prime[i*State_size+1]
            temp_sd[0][0] = S_dprime[i*State_size]
            temp_sd[1][0] = S_dprime[i*State_size+1]
            #//copy M*S_prime for the matrix calculation of m.  These multiplications are performed using BLAS routines.  7page
            #scipy.linalg.blas.dgemv(chn, m_m, n_m, one, M_dynamics[i*State_size*State_size], m_m, S_prime[i*State_size], p_m, zero, datam, p_m)
            #print(i)
            #print(len(S_prime[i*State_size]))
            #print(M_dynamics[i*State_size*State_size])
            #print(temp_m, temp_s)
            #fblas.dgemv(one,temp_m, temp_s,zero,temp_d,overwrite_y = 1)#,m_m,p_m,n_m,p_m, 0,0) #, m_m) #  chn, m_m, n_m, , , m_m,, p_m, zero, datam, p_m)
            #scipy.linalg.blas.dgemv(one, M_dynamics, S_prime ,zero,datam,m_m,p_m,n_m,p_m, chn, m_m)
            temp_dm = np.dot(temp_m,temp_s)
            datam = np.reshape(temp_dm,(2))
            #print(temp_dm)
            #print(datam)
            
            #//Perform M*S_dprime
            #scipy.linalg.blas.dgemv(chn, m_m, n_m, one, M_dynamics[i*State_size*State_size], m_m, S_dprime[i*State_size], p_m, zero, datac1, p_m)
            #fblas.dgemv(one, M_dynamics[i*State_size*State_size], S_dprime[i*State_size],zero,datac1,m_m,p_m,zero,p_m, chn,m_m)
            temp_dc1 = np.dot(temp_m,temp_sd)
            datac1 = np.reshape(temp_dc1,(2))
            #print(datac1)
            #print(datac1)
            #//C*S_prime^2;
            #scipy.linalg.blas.dgemv(chn, m_m, n_m, one, C_dynamics[i*State_size*State_size], m_m, S_prime[i*State_size], p_m, zero, datac2, p_m)
            #fblas.dgemv(one, C_dynamics[i*State_size*State_size], S_prime[i*State_size],zero,datac2,m_m,p_m,zero,p_m, chn,m_m)
            temp_dc2 = np.dot(temp_c,temp_s)
            datac2 = np.reshape(temp_dc2,(2))
            #print(datac2)
            #//Now perform all of the assignment;
            #//not these indices are all one less than those in matlab;
            if(i==0):
                for j in range(State_size):
                    A_i[j] = j #;//c/2
                    A_j[j] = 0
                    A_v[j] = (datac1[j]+datac2[j])/2.0
                    A_i[State_size+j] = j #;//m
                    A_j[State_size+j] = 1
                    #print(datac1, datac2)
                #memcpy(&A_v[State_size], datam, State_size*sizeof(double));
                #memcpy(&A_i[2*State_size], indexR_i, U_size*State_size*sizeof(int));//-R
                A_v[State_size] = datam[0]
                A_v[State_size+1] = datam[1]
                A_i[2*State_size] = indexR_i[0]
                A_i[2*State_size+1] = indexR_i[1]
                A_i[2*State_size+2] = indexR_i[2]
                A_i[2*State_size+3] = indexR_i[3]
                for j in range(U_size*State_size):
                    A_j[2*State_size+j] = indexR_j[j]+2
                    A_v[2*State_size+j] = R_dynamics[j]*(-1.0)
                    #print(R_dynamics[j] *(-1))
                
                A_i[State_size*(2+U_size)]= State_size #;//for the -1;
                A_j[State_size*(2+U_size)]= 0
                A_v[State_size*(2+U_size)]= -1.0
                A_i[State_size*(2+U_size)+1]= State_size #;//for the 2*dtheta
                A_j[State_size*(2+U_size)+1]= 1
                A_v[State_size*(2+U_size)+1]= 2*dtheta
                b_Ax[State_size]=(-1.0)*b_0
            else:
                for j in range(State_size):
                    A_i[i*block_length-State_size+j]= j+i*(State_size+1) #;//c/2
                    A_j[i*block_length-State_size+j]= i*(2+U_size)
                    A_v[i*block_length-State_size+j]= (datac1[j]+datac2[j])/2.0
                    A_i[i*block_length+State_size*(1+U_size)+2+j] = j+i*(State_size+1) #;//c/2 for the split
                    A_j[i*block_length+State_size*(1+U_size)+2+j] = (i-1)*(2+U_size)
                    A_v[i*block_length+State_size*(1+U_size)+2+j] = (datac1[j]+datac2[j])/2.0
                    A_i[i*block_length+j] = j+i*(State_size+1) #;//m
                    A_j[i*block_length+j] = i*(2+U_size)+1
                
                #memcpy(&A_v[i*block_length], datam, State_size*sizeof(double));
                A_v[i*block_length] = datam[0]
                A_v[i*block_length+1] = datam[1]
                for j in range(U_size*State_size):
                    A_i[i*block_length+State_size+j] = indexR_i[j]+i*(State_size+1)
                    A_j[i*block_length+State_size+j] = indexR_j[j]+2+i*(2+U_size)
                    A_v[i*block_length+State_size+j] = R_dynamics[i*State_size*U_size+j]*(-1.0)
                
                A_i[i*block_length+State_size*(1+U_size)]= State_size+i*(State_size+1) #;//for the -1;
                A_j[i*block_length+State_size*(1+U_size)]= 0+i*(2+U_size)
                A_v[i*block_length+State_size*(1+U_size)]= -1.0
                A_i[i*block_length+State_size*(1+U_size)+1]= State_size+i*(State_size+1) #;//for the 2*dtheta (fig (44))
                A_j[i*block_length+State_size*(1+U_size)+1]= 1+i*(2+U_size)
                A_v[i*block_length+State_size*(1+U_size)+1]= 2*dtheta
                A_i[i*block_length+State_size*(2+U_size)+2]= State_size+i*(State_size+1)
                A_j[i*block_length+State_size*(2+U_size)+2]=(i-1)*(2+U_size)
                A_v[i*block_length+State_size*(2+U_size)+2]= 1
                b_Ax[i*(State_size+1)+State_size]=0.0
            
            for j in range(State_size):
                b_Ax[i*(State_size+1)+j] = d_dynamics[i*State_size+j]*(-1.0)#-1e-29+d_dynamics[i*State_size+j] #;//the d term
                #print(b_Ax[i*(State_size+1)+j] * -1, " ",end="")
                # print("data", datac2[j])
                if(i==0):
                    b_Ax[i*(State_size+1)+j]-=b_0*(datac1[j]+datac2[j])/2.0 #;//the c/2*b[0] term.
            
            
        
    
    
    
    if(status):
        #//make sure that the indexed arrays are returned in addition to the entire matrix;
        # print(A_v)
        #print(A_v, len(A_v))
        pA_i = A_i
        #print(pA_i)
        pA_j = A_j
        pA_v = A_v
        pb_Ax = b_Ax
        
        A.nzmax = block_length*(S_length-1)-State_size
        A.i = A_i
        #print(A_i)
        A.p = A_j
        A.x = A_v
        A.nz = A.nzmax
        # print("A.",A.x)
        A.m = (S_length-1)*(State_size+1)
        A.n = (S_length-1)*(2+U_size)
        pAc = csparse.cs_compress(A)
        #A.nz = -1
        #print("hihi")
        # print(pAc.x)
        #//cs_dupl(Ac);there shouldn't be any duplicates to remove;
        csparse.cs_dropzeros(pAc)
        #pAc = csparse.cs_dropzeros(A)
        #print("hihi")
        #pAc = A
       #print("paz",pAc.nz)
    
    #//free the arrays from the matrix multiply
    if(np.any(datam != None)):
        datam = []
    if(np.any(datac1 != None)):
        datac1= []
        
    if(np.any(datac2 != None)):
        datac2= []
        
    
    if(np.any(indexR_i != None)):
        indexR_i= []
    if(np.any(indexR_j != None)):
        indexR_j= []

    if(np.any(R_dynamics != None)):
        R_dynamics= []
    if(np.any(M_dynamics != None)):
        M_dynamics= []
    if(np.any(C_dynamics != None)):
        C_dynamics= []
    if(np.any(d_dynamics != None)):
        d_dynamics= []
    #print(pA_v)
    if(status):
        return index_length, pAc,  pA_i,  pA_j,  pA_v,  pb_Ax,  b_0
    else:
        return -1
    


#//modified from Tim Davis's routine
def so_lu_solve( A,  b,  S,  orderOkay):
    tol = 0.0
    #x = [] 
    #N = csparse.csn()
    order, n, ok = 0,0,0
    #print("asd",A.nz)
    
    tol = 1e-10
    order = 1
    
    if (not A or not b): return (0) #		/* check inputs */
    n = A.n
    # if cnt==1:
    #     for i in range(6003):
    #         print("A ",A.x[i] , end="")
    #print("1111111")
    if(not orderOkay):
        if(S!=None):
            S = None
            #csparse.cs_sfree(S)
        # A= csparse.cs_compress(A)
        #print("a",A.nz)#, len(A.nz) )
        S = csparse.cs_sqr (order, A, 0) #/* ordering and symbolic analysis */
        # for i in range(100):
        #     print(S.q[i], end = " ")
    
    #print(N)
    N = csparse.cs_lu (A, S, tol) #		/* numeric LU factorization */
    # for i in range(300):
    #     print(N.L.x[i], end=' ')
    # print("N",N.pinv[0], end=' ')
    #print("2222222")
    #print((N.pinv))
    #print("l",A)
    #print("LU",N.L, N.U)
    #x = csparse.cs_malloc (n, sizeof (double)) ;
    x = [None] * n
    ok = (S and N and x) 
    #print("123123")
    #print(" bbig ", b[1])
    
    if (ok):
    
        #csparse.cs_ipvec (n, N.pinv, b, x)# ;	/* x = P*b */
        csparse.cs_ipvec (N.pinv, b, x, n)# ;	/* x = P*b */
        # print("N",N.pinv, end=' ')
        # if cnt <5:
        #     print(x)
        # print(" x1x1 ",x[1])
        csparse.cs_lsolve (N.L, x)# ;		/* x = L\x */
        # if cnt <5:
        #     print(x)
        csparse.cs_usolve (N.U, x)# ;		/* x = U\x */
        # csparse.cs_ltsolve (N.U, x)# ;		/* x = U\x */
        # if cnt <5:
        #     print(x)
        #csparse.cs_ipvec (n, S.Q, x, b)# ;	/* b = Q*x */
        csparse.cs_ipvec (S.q, x, b, n)# ;	/* b = Q*x */
        # csparse.cs_pvec (S.q, x, b, n)# ;	/* b = Q*x */
        
        # if cnt <5:
        #     print(" b1b1 ",b)
        
    #print("234234")
    #print(" bbig2 ", b[1])
    #csparse.cs_free (x) 
    #csparse.cs_nfree (N) 
    x = None
    N = None

    # if not csparse.CS_CSC(A) or b == None:
    #     return False # check inputs
    # n = A.n
    # S = csparse.cs_sqr(order, A, False) # ordering and symbolic analysis
    # #print("S",S.q)
    # N = csparse.cs_lu(A, S, tol) # numeric LU factorization
    # #print(("N",N.pinv[0]))
    # x = csparse.xalloc(n) # get workspace
    # ok = S != None and N != None
    # if ok:
    #     csparse.cs_ipvec(N.pinv, b, x, n) # x = b(p)
    #     csparse.cs_lsolve(N.L, x) # x = L\x
    #     csparse.cs_usolve(N.U, x) # x = U\x
    #     csparse.cs_ipvec(S.q, x, b, n) # b(q) = x
    
    # x = None
    # N = None
    
    return ok,A,b,S




def so_stuffing( Abig_v,  Abig_i,  Abig_j,  indexH_v,  indexH_i,  indexH_j,  Hindex_length,  indexA21_v,  indexA21_i,  indexA21_j,  A21index_length,  Hc_m,  S_length,  State_size,  U_size):
    #i = 0
    
    #//copy the H array into the big array, but changing ordering to interweave dual varialbes.
    #memcpy(Abig_v, indexH_v, Hindex_length*sizeof(double));
    #memcpy(Abig_i, indexH_i, Hindex_length*sizeof(int));
    #memcpy(Abig_j, indexH_j, Hindex_length*sizeof(int));
    
    # Abig_v = indexH_v
    # Abig_i = indexH_i
    # Abig_j = indexH_j
    # for i in range(100):
    #     print(indexH_v[i], end="")
    for i in range(Hindex_length):
        Abig_v[i] = indexH_v[i]
        Abig_i[i] = indexH_i[i]
        Abig_j[i] = indexH_j[i]
    #print("stuff",len(Abig_v))
    #print(len(indexH_i))
    
    #//copy the A21 array into the big array;
    #memcpy(&Abig_v[Hindex_length], indexA21_v, A21index_length*sizeof(double));
    #memcpy(&Abig_v[Hindex_length+A21index_length], indexA21_v, A21index_length*sizeof(double));
    #memcpy(&Abig_j[Hindex_length], indexA21_j, A21index_length*sizeof(int));
    #memcpy(&Abig_i[Hindex_length+A21index_length], indexA21_j, A21index_length*sizeof(int));
    #print("d", Hindex_length, A21index_length)
    #print("dd", len(indexA21_v), len(indexA21_j))
    #print(len(Abig_v))
    # for i in range(A21index_length):
    #     Abig_v[Hindex_length+i] = indexA21_v[i]
    for i in range(A21index_length):
        Abig_v[Hindex_length+i] = indexA21_v[i]
        Abig_v[Hindex_length+A21index_length+i] = indexA21_v[i]
        Abig_j[Hindex_length+i] = indexA21_j[i]
        Abig_i[Hindex_length+A21index_length+i] = indexA21_j[i]
    # Abig_v[Hindex_length] = indexA21_v
    # Abig_v[Hindex_length+A21index_length] = indexA21_v
    # Abig_j[Hindex_length] = indexA21_j
    # Abig_i[Hindex_length+A21index_length] = indexA21_j

    for i in range(A21index_length):
        Abig_i[Hindex_length+i] = indexA21_i[i]+Hc_m
        Abig_j[Hindex_length+A21index_length+i] = indexA21_i[i]+Hc_m
    
    return Abig_v,  Abig_i,  Abig_j
