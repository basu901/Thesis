import numpy as np
import copy
import sys
from scipy.optimize import minimize,fmin_slsqp
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import utils as ut
from itertools import combinations

class LinearSystem(object):

    def __init__(self,n, m_arr, L, g, epsilon, k, prev_conn):
        self.n = n
        self.m_arr = m_arr
        self.L = L
        self.g = g
        self.epsilon = epsilon
        self.k = k
        self.prev_conn = prev_conn
        self.s_pos = np.zeros(n)
        
       
    
    
    def obj_func_L1(self, abc_arr, xy_target_arr,n,L,lamda):
        alpha_arr = abc_arr[0:n]*self.k + (1-abc_arr[0:n])*self.epsilon
        beta_arr = abc_arr[n:2*n+1]*self.k + (1-abc_arr[n:2*n+1])*self.epsilon
        gamma_arr = abc_arr[2*n+1:3*n+1]*self.k + (1-abc_arr[2*n+1:3*n+1])*self.epsilon

        
        eta = self.build_diagonally(n,self.prev_conn,abc_arr)

        #print(eta)
        
        xys_arr = self.find_xys_given_abc(N,alpha_arr,beta_arr,gamma_arr,eta,L)

        #print(xys_arr)

        xys_arr = xys_arr.reshape(n, 3)
        
        xy_arr = xys_arr[:,0:2]
        self.s_pos = xys_arr[:,2]
        xy_target_arr = xy_target_arr.reshape((n,2))
        '''x_target = xy_target_arr[:,0]
        y_target = xy_target_arr[:,1]

        x = np.zeros((n,))
        y = np.zeros((n,))
        s = np.zeros((n,))
        x[0] = xys_arr[0]
        y[0] = xys_arr[1]
        s[0] = xys_arr[2]
        for i in range(1, n):
            x[i] = xys_arr[i*3] + x[i-1]
            y[i] = xys_arr[i*3 + 1] + y[i-1]
            s[i] = xys_arr[i*3 + 2] + s[i-1]
            x_target[i] = x_target[i-1]+x_target[i]
            y_target[i] = y_target[i-1]+y_target[i]'''
        
        #print(xy_arr)
        #print xy_target_arr
        #print np.absolute(xy_target_arr - xy_arr)
        #error = np.sum(abs(xy_target_arr - xy_arr))


        
        #finding the error between distance between consecitive points
        error_1 = 0
        for i in range(len(xy_arr)-1):
            error_1 += abs(abs((xy_arr[i+1][0]-xy_arr[i][0])-(xy_target_arr[i+1][0]-xy_target_arr[i][0]))-abs((xy_arr[i+1][1]-xy_arr[i][1])-(xy_target_arr[i+1][1]-xy_target_arr[i][1])))
        error_2 = 0
        error_2 = np.sum(np.sum((xy_arr - xy_target_arr)**2, axis=0))

        error = lamda*error_2 + (1-lamda)*((error_1)**2)
        #else:
            #error = np.sum(np.sum((xy_target_arr - xy_arr)**2,axis = 1)) - lamda*np.sum(abc_arr[0:n]*np.log(abc_arr[0:n])) - lamda*np.sum(abc_arr[n:2*n+1]*np.log(abc_arr[n:2*n+1])) - lamda*np.sum(abc_arr[2*n+1:3*n+1]*np.log(abc_arr[2*n+1:3*n+1]))
        
        return error

    def get_s_pos(self):
        return self.s_pos
  
    def find_xys_given_abc(self,N,a,b,c,eta,L):
        x = ut.linear(N,a,b,c,eta,self.m_arr,L)
        return x


    
    #Used to construct the eta matrix diagonally from abc array
    def build_diagonally(self,N,conn,abc):
        eta = np.zeros(shape = (N,N))
        k = 3*N + 1
        itr = 0
        while(itr<conn):
            if itr == 0:
                i = itr+1
                j = i
            else:
                i = itr
                j = i -itr
            while(i<N):
                eta[i][j] = abc[k]*self.k+(1-abc[k])*self.epsilon
                i += 1
                j += 1
                k += 1
            itr += 1

        return eta



def obtained_desired(xy_target_arr,xys_or,xys_r,xys_rr,n):
    xy_target_arr = xy_target_arr.reshape((n,2))
    x_target = copy.deepcopy(xy_target_arr[:,0])
    y_target = copy.deepcopy(xy_target_arr[:,1])

    x_or = np.zeros((n,))
    y_or = np.zeros((n,))
    s_or = np.zeros((n,))
    x_or[0] = xys_or[0]
    y_or[0] = xys_or[1]
    s_or[0] = xys_or[2]


    x_r = np.zeros((n,))
    y_r = np.zeros((n,))
    s_r = np.zeros((n,))
    x_r[0] = xys_r[0]
    y_r[0] = xys_r[1]
    s_r[0] = xys_r[2]


    x_rr = np.zeros((n,))
    y_rr = np.zeros((n,))
    s_rr = np.zeros((n,))
    x_rr[0] = xys_rr[0]
    y_rr[0] = xys_rr[1]
    s_rr[0] = xys_rr[2]

    for i in range(1, n):
        x_or[i] = xys_or[i*3] + x_or[i-1]
        y_or[i] = xys_or[i*3 + 1] + y_or[i-1]
        s_or[i] = xys_or[i*3 + 2] + s_or[i-1]

    for i in range(1, n):
        x_r[i] = xys_r[i*3] + x_r[i-1]
        y_r[i] = xys_r[i*3 + 1] + y_r[i-1]
        s_r[i] = xys_r[i*3 + 2] + s_r[i-1]

    for i in range(1, n):
        x_rr[i] = xys_rr[i*3] + x_rr[i-1]
        y_rr[i] = xys_rr[i*3 + 1] + y_rr[i-1]
        s_rr[i] = xys_rr[i*3 + 2] + s_rr[i-1]
      
    for i in range(1, n):
        x_target[i] = x_target[i-1]+x_target[i]
        y_target[i] = y_target[i-1]+y_target[i]

    l0_fig = plt.figure()
    l0 = plt.subplot(111)
    zer = np.zeros((n,))
    l0.plot([0,L],[0,0],color = 'k',marker = 'o')

    l0.plot([0,x_or[0]],[0,y_or[0]], color = 'c',marker = 'o')
    l0.plot(x_or, y_or, color='c', marker='o')
    l0.scatter(s_or,zer,color = 'c', marker = 'x')

    
    l0.plot([0,x_r[0]],[0,y_r[0]], color = 'b',marker = 'o')
    l0.plot(x_r, y_r, color='b', marker='o')
    l0.scatter(s_r,zer,color = 'b', marker = 'x')

    l0.plot([0,x_rr[0]],[0,y_rr[0]], color = 'g',marker = 'o')
    l0.plot(x_rr, y_rr, color='g', marker='o')
    l0.scatter(s_rr,zer,color = 'g', marker = 'x')

    l0.scatter(x_target,y_target,color = 'r', marker = '^')
    
    l0.invert_yaxis()

    plt.show()

    del x_target,y_target




def add_plot_animation(pos,n,xy_target_arr,pic_index):
    #Used to generate individual plots to display as animation
    x= np.zeros(n)
    y= np.zeros(n)
    s= np.zeros(n)
    zer = np.zeros(n)
    x[0] = pos[0]
    y[0] = pos[1]
    s[0] = pos[2]

    fig = plt.figure()
    
    for i in range(1, n):
        x[i] = pos[i*3] + x[i-1]
        y[i] = pos[i*3 + 1] + y[i-1]
        s[i] = pos[i*3 + 2] + s[i-1]

    plt.plot([0,x[0]],[0,y[0]], color = 'b',marker = 'o')
    plt.plot(x, y, color='b', marker='o')
    plt.scatter(s,zer,color = 'c', marker = 'x')
        

    xy_target_arr = xy_target_arr.reshape((n,2))
    x_target = copy.deepcopy(xy_target_arr[:,0])
    y_target = copy.deepcopy(xy_target_arr[:,1])

    for i in range(1, n):
        x_target[i] = x_target[i-1]+x_target[i]
        y_target[i] = y_target[i-1]+y_target[i]

    plt.plot([0,L],[0,0],color = 'k',marker = 'o')
    plt.scatter(x_target,y_target,color = 'r',marker = '^')
    plt.gca().invert_yaxis()
    plt.xlabel("X location")
    plt.ylabel("Y location")
    loc = 'plot'+str(pic_index)+'.png'
    fig.savefig(loc)
    plt.close(fig)

    


def add_plot(pos,n,xy_target_arr):
    #Used to display all configurations together
    for j in range(len(pos)):
        x= np.zeros(n)
        y= np.zeros(n)
        s= np.zeros(n)
        zer = np.zeros(n)
        x[0] = pos[j][0]
        y[0] = pos[j][1]
        s[0] = pos[j][2]

        for i in range(1, n):
            x[i] = pos[j][i*3] + x[i-1]
            y[i] = pos[j][i*3 + 1] + y[i-1]
            s[i] = pos[j][i*3 + 2] + s[i-1]

        plt.plot([0,x[0]],[0,y[0]], color = 'b',marker = 'o')
        plt.plot(x, y, color='b', marker='o')
        plt.scatter(s,zer,color = 'c', marker = 'x')
        

    xy_target_arr = xy_target_arr.reshape((n,2))
    x_target = copy.deepcopy(xy_target_arr[:,0])
    y_target = copy.deepcopy(xy_target_arr[:,1])

    for i in range(1, n):
        x_target[i] = x_target[i-1]+x_target[i]
        y_target[i] = y_target[i-1]+y_target[i]

    plt.plot([0,L],[0,0],color = 'k',marker = 'o')
    plt.scatter(x_target,y_target,color = 'r',marker = '^')
    plt.gca().invert_yaxis()
    plt.show()




def modify(x,comb_r,prev_conn,N,k,epsilon,ls,xy_target_arr):
    
    tot_eta = ((N*(N+1))/2) - (((N-prev_conn)*(N-prev_conn+1))/2) - 1
    comb_n = int(3*N+1+tot_eta)

    comb_n_list = list(range(comb_n+1))
    comb_n_list.pop(0)

    pos_grid = list()

    comb = combinations(comb_n_list,comb_r)

    num1 = 0
    
    num2 = 0
    
    num3 = 0

    count = 0
    
    for i in comb:

        if comb_r==1:
            num1 = i[0]-1

        if comb_r == 2:
            num1 = i[0]-1
            num2 = i[1]-1

        if comb_r == 3:
            num1 = i[0]-1
            num2 = i[1]-1
            num3 = i[2]-1


        if num1>0:
            x[num1]= 1 - x[num1]
        if num2>0:  
            x[num2] = 1 - x[num2]
        if num3>0:
            x[num3] = 1 - x[num3]


        alpha = x[0:N]*k + (1-x[0:N])*epsilon
        beta = x[N:2*N+1]*k + (1-x[N:2*N+1])*epsilon
        gamma = x[2*N+1:3*N+1]*k + (1-x[2*N+1:3*N+1])*epsilon

        eta = ls.build_diagonally(N,prev_conn,x)

        print("HERE")

        X = ut.linear(N,alpha,beta,gamma,eta,mass,L)
        print("Values of a: ",x[0:N])
        print("Values of b: ",x[N:2*N+1])
        print("Values of c: ",x[2*N+1:3*N+1])
        print("Eta_grid:", eta)
 
        add_plot_animation(X,N,xy_target_arr,count)    
        

        count += 1

        #Reversing to original value
        if num1>0:
            x[num1]= 1 - x[num1]
        if num2>0:
            x[num2] = 1 - x[num2]
        if num3>0:
            x[num3] = 1 - x[num3]
        

        del X
        
    

prev_conn = 4
N = 10
mass = np.ones(N)
epsilon = 0.1
k = 2

eta = np.zeros(shape=(N,N))

tot_eta = int(((N*(N+1))/2) - (((N-prev_conn)*(N-prev_conn+1))/2) - 1)

L =50
g = 9.8

ls = LinearSystem(N, mass, L, g, epsilon, k, prev_conn)

abc_arr = np.zeros(3*N+1+tot_eta)+0.1

bound_spring = np.zeros((3*N+1+tot_eta,2))
bound_spring[:,1] = 1

alphas = abc_arr[0:N]*epsilon + (1-abc_arr[0:N])*k
beta_arr = abc_arr[N:2*N+1]*epsilon + (1-abc_arr[N:2*N+1])*k
gamma_arr = abc_arr[2*N+1:3*N+1]*epsilon + (1-abc_arr[2*N+1:3*N+1])*k



#creating constraint method
'''def s_constraint(self):
    x = np.asarray([0 if i>0 else 1 for i in ls.get_s_pos()])
    return x.sum()

my_constraints = ({'type':'eq',"fun":s_constraint})'''



xy_target_arr = np.array([0.93399589,8.50121861,1.27033057,2.94673055,2.21326811,0.14332102,
 3.17328161, 0.3535583,  4.51861064, 1.22969449, 0.93399589, 8.50121861,
 1.27033057, 2.94673055, 2.21326811, 0.14332102, 3.17328161, 0.3535583,
 4.51861064, 1.22969449])


print("Provided (x,y) values:")
print(xy_target_arr)

lamda = [1]
error = np.zeros(len(lamda))

for i in range(len(lamda)):
    res = minimize(ls.obj_func_L1, abc_arr, args=(xy_target_arr,N,L,lamda[i]),
                                              method = "trust-constr",bounds = bound_spring,options={'disp': True})
    
    error[i] = res.fun

minimum_error = np.amin(error)

for j in range(len(error)):
    if error[j]==minimum_error:
        break


res = minimize(ls.obj_func_L1, abc_arr, args=(xy_target_arr,N,L,lamda[j]),
                                              method = "trust-constr",bounds = bound_spring,options={'disp': True})
print("Function Error Value:",res.fun)
print("Lagrangian gradient value:", res.lagrangian_grad)
print("Gradient Value:", res.grad)

print("tr_radius:", res.tr_radius)


alpha_arr = res.x[0:N]*k + (1-res.x[0:N])*epsilon
beta_arr = res.x[N:2*N+1]*k + (1-res.x[N:2*N+1])*epsilon
gamma_arr = res.x[2*N+1:3*N+1]*k + (1-res.x[2*N+1:3*N+1])*epsilon

eta = ls.build_diagonally(N,prev_conn,res.x)

xys_or = ut.linear(N,alpha_arr,beta_arr,gamma_arr,eta,mass,L)

print("OBTAINED:",xys_or)

add_plot_animation(xys_or,N,xy_target_arr,0)

print("Value of lambda:",lamda[j])
print("Values given by optimizer:")
print("alphas = ",alpha_arr)
print("betas = ",beta_arr)
print("gammas = ",gamma_arr)
print("eta matrix = ",eta)

'''
for i in range(0,len(res.x)):
    if res.x[i]>=0.5:
        res.x[i] = 1
    else:
        res.x[i] = 0

alpha_arr = res.x[0:N]*k + (1-res.x[0:N])*epsilon
beta_arr = res.x[N:2*N+1]*k + (1-res.x[N:2*N+1])*epsilon
gamma_arr = res.x[2*N+1:3*N+1]*k + (1-res.x[2*N+1:3*N+1])*epsilon

eta = ls.build_diagonally(N,prev_conn,res.x)

print("Values after rounding:")
print("alphas = ",alpha_arr)
print("betas = ",beta_arr)
print("gammas = ",gamma_arr)
print("eta matrix = ",eta)

xys_r = ut.linear(N,alpha_arr,beta_arr,gamma_arr,eta,mass,L)

#randomly modify the spring stiffness'
#Forming the list containing the list of index values for non zero eta elements.
grid_idx_list = list()

for i in range(N-1):
    count = 0
    for j in range(i,N):
        if count<prev_conn:
           grid_idx_list.append([N-i-1,N-j-1])
           count = count+1


for i in range(1,2):
    positions = modify(res.x,i,prev_conn,N,k,epsilon,ls,xy_target_arr)
    

res.x = 1-res.x


alpha_arr = res.x[0:N]*k + (1-res.x[0:N])*epsilon
beta_arr = res.x[N:2*N+1]*k + (1-res.x[N:2*N+1])*epsilon
gamma_arr = res.x[2*N+1:3*N+1]*k + (1-res.x[2*N+1:3*N+1])*epsilon

eta = ls.build_diagonally(N,prev_conn,res.x)

print("Values after reverse rounding:")
print("alphas = ",alpha_arr)
print("betas = ",beta_arr)
print("gammas = ",gamma_arr)
print("eta matrix = ",eta)

xys_rr = ut.linear(N,alpha_arr,beta_arr,gamma_arr,eta,mass,L)

obtained_desired(xy_target_arr,xys_or,xys_r,xys_rr,N)'''







