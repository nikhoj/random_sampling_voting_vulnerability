'''

This code is generated for the paper 
'How Hard is Bribery in Elections with Randomly Selected Voters'
Last Updated: 6/23/2021


'''

import numpy as np
import random
import math
import operator as op
from functools import reduce
from gurobipy import *
import timeit
#start = timeit.default_timer()

#global variable declaration here

global n
global m
global B
global p
global EPS
global tau
global vote_count
global lamda
global max_V_i
global F_container
global min_bcost
global max_bcost

#parse your information here

m = 3               #number of candidate (excluding the designated candidate)
n = 100             #number of voters including the voters of the designated candidate
B = 200000          #total bribing budget
EPS =[.5,.6,.7,.8]  #epsilon
p = 0.5             #probability to get selected in committee
min_bcost = 50      #minimum bribe cost for a voter
max_bcost = 100     #maximmum bribe cost for a voter

#function to calculate the combination nCr formula
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def bribing_instance_generator(min_bcost,max_bcost):
     
    votes = np.zeros((n,m+1))
    vote_count = np.sum(votes, axis = 0)

    while np.amax(vote_count, axis = 0) == vote_count[m]:
        votes = np.zeros((n,m+1))
        bribe_cost = np.zeros((n,m))
        for i in range(n):
            rnd_vote = random.randint(0, m)
            votes[i,rnd_vote] = 1
            if rnd_vote != m:
                rnd_cost = random.randint(min_bcost, max_bcost)
                bribe_cost[i,rnd_vote] = rnd_cost
            
        vote_count = np.sum(votes, axis = 0)
            
    lamda = np.zeros((int(max(vote_count)+1),m))
    for i in range(m):
        temp = bribe_cost[:,i]
        temp = temp.ravel()[np.flatnonzero(temp)]
        temp = np.cumsum(temp)
        lamda[1:len(temp)+1,i] = temp  
    
    return vote_count, lamda        

#vote_count, lamda = bribing_instance_generator(min_bcost,max_bcost)
#max_V_i = lamda.shape[0] #maximun votes count
#tau = int(np.floor(1/(np.log(1 + eps/4)))) 
#F_container = np.zeros((m*max_V_i + 1 + tau),dtype =int )




#This function will generate unique value for s_star in ascending manner
def S_star_generator():
    
    global tau
    global n
    
    S_star = np.sort(np.random.choice(range(1,n), size=(tau)))
    while len(S_star) != len(np.unique(S_star)):
        S_star = np.sort(np.random.choice(range(1,n), size=(tau)))
    
    return S_star

#S_star = S_star_generator()

#Some Auxiliary function
def small_phi(y_i, t):
    
    global p
    
    if t <= y_i:
        return ncr(y_i,t)*math.pow(p,t)*math.pow(1-p,y_i-t)
    else:
        return 0

def big_phi(k, S, t):
    
    global p
    
    ret = 0
    for h in range(S[t]):
        ret += small_phi(k,h)
    
    return ret

def big_phi2(k, St):
    
    global p
    
    ret = 0
    for h in St:
        ret += small_phi(k,h)
    
    return ret

def fkt_matrix_generator(S_star):
    
    global max_V_i
    global tau
    global m
    global n
    
    fkt = np.zeros((tau,max_V_i))
    for k in range(0,max_V_i):
        
        for t in range(0,tau):
            ret = big_phi(k, S_star, t)                
            fkt[t,k] = np.floor(m*n*tau*np.log(ret))
            #print("when k is " + str(k) + ", t is " + str(t) + ", and S[t] is " + str(S_star[t]) +", then big_phi = " + str(ret) + ", and fkt = " + str(fkt[t,k]))
    print(fkt)                    
    return fkt

#fkt_matrix = fkt_matrix_generator(S_star)
          
#RIP algorithm
def run_RIP(r, S_star):
    
    global F_container
    global max_V_i
    global m
    
    fkt_matrix = fkt_matrix_generator(S_star)
    
    I = m
    K = max_V_i
    
    # Model
    model = Model("BribingOPT")
    
    # Create decision variables
    x = model.addVars(I,K , vtype= GRB.BINARY, name="x")
    
    # Set objective 
    model.setObjective(quicksum(quicksum(x[i,k]*lamda[k,i] for k in range(K)) for i in range(I)), GRB.MINIMIZE)
    
    
    #constraint 2a
    model.addConstr((quicksum(quicksum(k* x[i, k] for i in range(I) ) for k in range(K)) == n-r ), name="Constraint 2a")
    
    #constraint 2b
    model.addConstrs((quicksum(quicksum(fkt_matrix[t,k]*x[i,k] for k in range(K)) for i in range(I)) <= m*n*tau*np.log(t+1/tau)) for t in range(tau))
    
    #Constraints 2c
    model.addConstrs((quicksum(x[i,k] for k in range(0,int(vote_count[i]+1))) == 1) for i in range(I))
    model.addConstrs((quicksum(x[i,k] for k in range(int(vote_count[i]+1),K)) == 0)for i in range(I))
    
    
    try:
        # Optimize model
        model.optimize()
        
        if model.SolCount == 0:
            print('No solution found, optimization status = %d' % model.Status)
        else:
            current_sol = []
            
            obj = model.getObjective()
            #print("Number of voters get bribed = " + str(n-r))
            #print("Total bribing cost for this was $" + str(obj.getValue()))
            
            #x_val = []
            for v in model.getVars():
                #print('%s %g' % (v.varName, v.x))
                current_sol.append(v.x)
            
            #x_val = np.asarray(x_val, int)
            #x_val = np.reshape(x_val, (I,K))
            
            #print(x_val)
            
            current_sol.append(n-r)
            #current_sol.append(S_star)
            current_sol = np.asarray(current_sol, int)
            current_sol = np.concatenate((current_sol,S_star))
            #print(current_sol)
            
            if obj.getValue() <= B:
                #print(len(current_sol))
                F_container = np.vstack((F_container, current_sol))
                #print(len(F_container))
                
                #return F_container
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')

'''
for indx in range(1,100):
    lamda = bribing_instance_generator(min_bcost,max_bcost)
    np.savetxt("Zaman_Data/lamda_n_300_m_10_mincost_50_maxcost_100/lamda_n_300_m_10_mincost_50_maxcost_100_{}.txt".format(indx), lamda)


check = np.loadtxt("lamda_100_3_08_05_50_100.txt", int)
vote_count = np.count_nonzero(check,0)
'''
#run_RIP(3,S_star)

def main_Algo():
    global n
    global m
    global B
    global p
    global EPS
    global tau
    global vote_count
    global lamda
    global max_V_i
    global F_container
    global min_bcost
    global max_bcost
    
    
    final_table = []
    final_runtime = []
    for eps in EPS:
        
        for i in range(1,100):
        
            time = 0
            cnt = 0
            lamda = np.loadtxt("Zaman_Data/lamda_n_{}_m_{}_mincost_50_maxcost_100/lamda_n_{}_m_{}_mincost_50_maxcost_100_{}.txt".format(n,m,n,m,i), int)
            vote_count = np.count_nonzero(lamda,0)
            max_V_i = lamda.shape[0] #maximun votes count
            tau = int(np.floor(1/(np.log(1 + eps/4)))) 
            F_container = np.zeros((m*max_V_i + 1 + tau),dtype =int ) #F_container generate
            for r in range (n-int(max(vote_count)),n):
                
                for count in range(100):      #**********set here how many S_star you want to try********
                    S_star = S_star_generator()
                    start = timeit.default_timer()
                    run_RIP(n-r,S_star)
                    stop = timeit.default_timer()
                    time = time + (stop - start)
                    cnt = cnt + 1
            #calculate the last column to find the best result
            try:
                #print('Average Time: ', time/cnt)
                avg_time = (time/cnt) * 1000
                temp = np.zeros((len(F_container),1),dtype=int)
                F_container = np.concatenate((F_container,temp),1)
                F_container = F_container[1:,:]
                        
                S = F_container[:,-1*tau -1:-1]
                
                #print("Here")
                zt_array = np.zeros((F_container.shape[0], 1))
                zt_array = np.concatenate((S,zt_array),1)
                
                
                for i in range(len(zt_array)):
                    for t in range(tau):
                        for k in range(max_V_i):
                            zt_array[i,5] += small_phi(r,t)*np.exp(np.log(big_phi(k,S[i],t))*F_container[i,k])
                #print("Here")           
                zt_array = np.reshape(zt_array[:,5],  (len(zt_array[:,5]),1))
                F_container = np.concatenate((F_container,zt_array),1)
                index = np.argmax(F_container[:,-1],0)
                #print("Here")
                 
                X = F_container[index, :max_V_i*m]
                X = np.reshape(X,(m,max_V_i))
                cost_total = np.sum(np.dot(X ,lamda))
                
                final_table.append(cost_total)
                final_runtime.append(avg_time)
                #print(X)
                #print("Total bribing cost is $" +str(cost_total))
            #return X
            except:
                print("failed")
                #return 0
    return final_table, final_runtime      
    
ret_val, ret_runtime = main_Algo()
ret_val = np.reshape(np.array(ret_val),(4,99))
ret_runtime = np.resize(np.array(ret_runtime),(4,99))
print(ret_val)     
print(ret_runtime)
