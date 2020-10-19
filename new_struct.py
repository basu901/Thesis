import numpy as np
import show_trunk
import copy,math
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from itertools import combinations
import revised_winding
import utils as ut

#Function used to calculate the effective stiffness in the max curl
def effective_stiffness(start,end,alpha):
    #print("ALPHA:",alpha)
    if not(start==0):
        start = start - 1
    i = start
    prod = 1
    alpha_sum = 0
    while i<=end:
        prod *= alpha[i]
        alpha_sum += alpha[i]
        i += 1
    result = prod/alpha_sum
    #print("STIFF:",result)
    return (result)



#Function used for Horizontal X Value Equilibrium
def generate_eta_sum_x(eta,n):
    eta_sum_x = np.zeros(n)
    for i in range(2,n+1):
        j = 0
        while j<i:
            eta_sum_x[i-1] += eta[i-1][j]
            j += 1
    return eta_sum_x


#Function used for Horizontal X Value Equilibrium
def generate_eta_sum_s(eta,n):
    eta_sum_s = np.zeros(shape=(n,n))
    for i in range(n-1,0,-1):
        j = i-1
        while j>-1:
            k = i
            l = i
            total = i - j
            count = 0
            while count<total:
                eta_sum_s[i][j] += eta[k][l]
                l -= 1
                count = count + 1
            j = j-1

    return eta_sum_s


#Function used for Horizontal Equilibrium for S masses
def generate_x_sum(eta,n):
    x_sum_s = np.zeros(n)
    count = 1
    k = 0
    i = 0
    for j in range(1,n):
        k = count
        while(k<n):
            x_sum_s[i] += eta[k][j]
            k = k+1
        count += 1
        i += 1
    return x_sum_s





#Used in the equilibrium of s masses        
def generate_x_sums_after_beta(n,eta,col,row):
    sum_x = 0
    while(row<n):
        if not (eta[row][col]==0):
            sum_x += eta[row][col]
            row += 1
        else:
            break
    return sum_x





def generate_y_sums(n,eta,k):
    col = k
    sum_y = 0
    while(col>=0):
        sum_y += eta[k][col]
        col -= 1
    #print(sum_y)
    return sum_y

def linear(N,a,b,c,eta,mass,L):
    n=N*3
    grid=np.zeros(shape=(n,n))
    val = np.zeros(n)
    k=0
    m=0
    g= 9.8

    eta_sum_x = np.zeros(N)
    eta_sum_s = np.zeros(shape = (N,N))

    eta_sum_x = generate_eta_sum_x(eta,N)
    eta_sum_x *= -1

    #print("Eta_Sum_x ", eta_sum_x)
    
    eta_sum_s = generate_eta_sum_s(eta,N)

    #print("Eta_Sum_s ", eta_sum_s)
    
    while(m<n):
        l=0 #Used to change sign of gamma
        j=0 #Keeping track of the column index
        count = 0
        while(j<n):
            if m==j:
                grid[m][j]=-(a[k]+c[k])+eta_sum_x[k]
                grid[m][j+2]=c[k]
                if j+3<n:
                    grid[m][j+3]=a[k+1]                
                break
            else:
                grid[m][j]+=c[k]*((-1)**(l+1))
                if l==0:
                    grid[m][j] += eta_sum_x[k]
                    l=1
                else:
                    grid[m][j] += eta_sum_s[k][count]
                    count += 1
                    j=j-1
                    l=0
            j=j+2
        k=k+1
        m=m+3

    #print("On adding x values: ", grid)

    x_sum_s = np.zeros(N)
    x_sum_s = generate_x_sum(eta,N)

    m = 2
    k =0
    
    while(m<n):
        l=0
        j=0
        while(j<n):
            if m==j:
                grid[m][j]=-(b[k]+c[k]+x_sum_s[k])
                if j+3<n:
                    grid[m][j+3]=b[k+1]
                sum_x = 0
                idx = j+1
                row = k+1
                while(idx<n):
                    col = k+1
                    #print("Row is:",row)
                    if(col<N):
                        sum_x = generate_x_sums_after_beta(N,eta,col,row)
                        grid[m][idx] = sum_x
                    idx += 3
                    row += 1
                break
                    
            else:
                grid[m][j]=(c[k]+x_sum_s[k])*((-1)**l)
                if l==0:
                    l=1
                else:
                    j=j-1
                    l=0
                
            j=j+2
        k=k+1
        m=m+3

    m=n-1
    j=2
    while j<n:
        grid[m][j]=grid[m][j]-b[N]
        j=j+3
    #print("On adding z values: ", grid)

    m=1
    k=0
    while m<n:
        j=1
        while j<n:
            if m==j:
                grid[m][j]=a[k]+c[k]+generate_y_sums(N,eta,k)
                if j+3<n:
                    grid[m][j+3]=a[k+1]*-1
                break
            else:
                grid[m][j]=c[k]+generate_y_sums(N,eta,k)
                j=j+2
            j=j+1
        m = m+3
        k=k+1

    #print("On adding y terms")  
    #print(grid)

    i = 1
    j = 0
    while i < n:
        val[i] = mass[j]*g
        i = i + 3
        j = j + 1
    
    val[-1]=-1*b[-1]*L           

    #print("Adding constants")
    #print(val)

    LU = linalg.lu_factor(grid)
    
    x = linalg.lu_solve(LU,val)

    return x




#Function used to generate mean and variance plots on the grid:
# x and y are the various end-effector co-ordinates
def grid_show(x,y,L):
    x_max = max(x) #max x co-ordinate of the end-effector positions
    x_min = min(x) #min x co-ordinate of the end-effector positions

    y_max = max(y) #max y co-ordinate of the end-effector positions
    y_min = min(y) #min y co-ordinate of the end-effector positions
    
    x_range = 2 #distance between x intervals on grid
    y_range = 2 #distance between y intervals on grid
    
    x_grid_high = x_max+(x_range-x_max%x_range)+x_range #max x grid line on grid
    x_grid_low = x_min - (x_min%x_range)-x_range        #min ' ' '..

    y_grid_high = y_max+(y_range-y_max%y_range)+y_range #max y grid line on grid
    y_grid_low = y_min - (y_min%y_range) - y_range      #min ' ' ' '....

    fig = plt.figure()
    ax = fig.gca()
    x_bounds = np.arange(x_grid_low, x_grid_high, x_range) #Used to make the grid boundaries on the X axis
    y_bounds = np.arange(y_grid_low, y_grid_high, y_range) #Used to make the grid boundaries on the Y axis

    x_mean = np.zeros(len(x_bounds)-1)
    y_mean = np.zeros(len(x_bounds)-1)

    x_var = np.zeros(len(x_bounds)-1)
    y_var = np.zeros(len(x_bounds)-1)

    grid_element_count = np.zeros(len(x_bounds)-1) #Used to store the number of elements in each grid block 

    for i in range(len(x)):
        for j in range(len(x_bounds)-1):
            if x_bounds[j]<=x[i]<x_bounds[j+1]:
                x_mean[j] += x[i]
                y_mean[j] += y[i]
                grid_element_count[j] += 1

    #print("x_total:",x_mean)
    #print("y_total:",y_mean)

    for i in range(len(x_mean)):
        if grid_element_count[i] > 0:
            x_mean[i] = x_mean[i]/grid_element_count[i]
            y_mean[i] = y_mean[i]/grid_element_count[i]

    #print("grid_count",grid_element_count)

    for i in range(len(x)):
        for j in range(len(x_bounds)-1):
            if x_bounds[j]<=x[i]<x_bounds[j+1]:
                x_var[j] += (x[i]-x_mean[j])**2
                y_var[j] += (y[i]-y_mean[j])**2

    for i in range(len(x_mean)):
        if grid_element_count[i] > 0:
            x_var[i] = x_var[i]/grid_element_count[i]
            y_var[i] = y_var[i]/grid_element_count[i]

    #print("Variance of y:",y_var)

    #x_sd = np.zeros(len(x_var))
    #y_sd = np.zeros(len(x_var))
    x_sd = np.sqrt(x_var) #storing the standard deviation
    y_sd = np.sqrt(y_var) 

    plt.scatter(x,y,color = 'r', marker = 'o')
    plt.scatter(x_mean,y_mean,color = 'k',marker = 'o',label='Mean')
    for i in range(len(x_sd)):
        plt.plot([x_mean[i]+x_sd[i],x_mean[i]-x_sd[i]],[y_mean[i]+y_sd[i],y_mean[i]-y_sd[i]],color = 'b')
    
    plt.plot([0,L],[0,0],color = 'k')
    ax.set_xticks(x_bounds)
    ax.set_yticks(y_bounds)
    ax.set_ylabel("Y location")
    ax.set_xlabel("X location")
    ax.set_title("Positional Statistic")
    ax.legend(loc = 'best')
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()
    



def check_positions(X,N):
    x = copy.deepcopy(X)
    x = x.reshape(N,3)
    x = x[:,0:2].T

    a = x[0]<0
    b = x[1]<0
    if np.dot(a,b)==True:
        return 4

    a = x[0]<0
    b = x[1]>0
    if np.dot(a,b)==True:
        return 2

    a = x[0]>0
    b = x[1]<0
    if np.dot(a,b)==True:
        return 3

    a = x[0]>0
    b = x[1]>0
    if np.dot(a,b)==True:
        return 1

    return 0



'''def classifier(X,N):
    x = copy.deepcopy(X)
    x = x.reshape(N,3)
    x = x[:,0:2]
    class_num = 0
    for i in range(0,N-1):
        print (math.atan2(x[i][1],x[i][0])-math.atan2(x[i+1][1],x[i+1][0]))
        if abs(math.atan2(x[i][1],x[i][0])-math.atan2(x[i+1][1],x[i+1][0]))>=(math.pi/4):
            class_num += 1

    del x
    print('\n')
    return class_num'''

#Used to calculate the final end_effector position of the structure and also returns the last delta x and delta y
def store_xy(X,n):
    i=0
    j=1
    k=2
    x = list()
    y=list()
    s=list()
    while i<n:
        x.append(X[i])
        y.append(X[j])
        s.append(X[k])
        i = i+3
        j = j+3
        k = k+3

    x_last = x[-1]
    y_last = y[-1]

    for i in range(1,len(x)):
        x[i] = x[i]+x[i-1]
        y[i] = y[i]+y[i-1]
        s[i] = s[i]+s[i-1]

    return x[-1],y[-1],x_last,y_last


'''
#N is the total number of masses on the line of masses
#conn is the number of previous connections
def diagonal_eta(N,conn):
    eta = np.zeros(shape=(N,N))
    itr = 0
    while(itr<conn):
        i = itr
        while(i<N):
            j = i-itr
            if not i==0 or not j== 0:
                eta[i][j]=np.random.uniform(low = 0.1, high = 2)
            i += 1
        itr += 1
    return eta


def diagonal_identity(N,conn):
    eta = np.zeros(shape=(N,N))
    itr = 0
    while(itr<conn):
        i = itr
        while(i<N):
            j = i-itr
            if not i==0 or not j== 0:
                eta[i][j]=1
            i += 1
        itr += 1
    return eta'''


#n is the total number of masses on the line of masses
# c is the number of previous connections
def make_eta(n,c,eps):
    grid = np.zeros(shape =(n,n))
    for i in range(n-1):
        count = 0
        for j in range(i,n):
            if count<c:
                grid[n-i-1][n-j-1] = eps
            count = count+1
    return grid
    



prev_conn = 4
N = 10
mass = np.ones(N)
epsilon =  0.1 #50 #0.0507N/mm
k =  2 #700 #0.719N/mm

eta = np.zeros(shape=(N,N))
eta = make_eta(N,prev_conn,epsilon)

L =50

x1_val = list()
y1_val = list()

x2_val = list()
y2_val = list()

x3_val = list()
y3_val = list()

x4_val = list()
y4_val = list()

x_info_anti = list()
y_info_anti = list()

x_info_clock = list()
y_info_clock = list()

clock_winding = list() #used to store the counter clockwise winding angles
anti_winding = list() #used to store the anti-clockwise winding angles
stiff = list() #used to store the effective stiffness in the region of maximum curl

a = np.zeros(N)
b = np.zeros(N+1)
c = np.zeros(N)

alpha = (1-a)*epsilon + a*k
beta = (1-b)*epsilon + b*k
gamma = (1-c)*epsilon + c*k        
X = linear(N,alpha,beta,gamma,eta,mass,L)

end_eff = ut.end_effector(X,N)
cm_w_anti,cm_w_clock,start,end = revised_winding.winding(X,N)
x_info_clock.append(end_eff[0]+0.5)
y_info_clock.append(end_eff[1]+0.5)
clock_winding.append(cm_w_clock)

x_info_anti.append(end_eff[0]-0.5)
y_info_anti.append(end_eff[1]-0.5)
anti_winding.append(cm_w_anti)

             
#For calculating the winding number of the entire trunk and effective stiffness:
#cm_w,start,end = revised_winding.winding(X,N)


#For showing the max clockwise and anti-clockwise curls:


stiffness = effective_stiffness(start,end,alpha)

#Forming the list containing the list of index values for non zero eta elements.
grid_idx_list = list()

for i in range(N-1):
    count = 0
    for j in range(i,N):
        if count<prev_conn:
           grid_idx_list.append([N-i-1,N-j-1])
           count = count+1

print(grid_idx_list)

tot_eta = ((N*(N+1))/2) - (((N-prev_conn)*(N-prev_conn+1))/2) - 1
comb_n = int(3*N+1+tot_eta)

comb_n_list = list(range(comb_n+1))
comb_n_list.pop(0)

comb_r = 1

count = 1


while(comb_r<5):
    comb = combinations(comb_n_list,comb_r)

    eta_affected_1 = False
    eta_affected_2 = False
    eta_affected_3 = False
    eta_affected_4 = False
    eta_affected_5 = False
    

    num1 = 0
    num2 = 0
    num3 = 0
    num4 = 0
    num5 = 0

    for i in list(comb):

        a = np.zeros(N)
        b = np.zeros(N+1)
        c = np.zeros(N)

        if comb_r==1:
            num1 = i[0]

        if comb_r == 2:
            num1 = i[0]
            num2 = i[1]

        if comb_r == 3:
            num1 = i[0]
            num2 = i[1]
            num3 = i[2]


        if comb_r == 4:
            num1 = i[0]
            num2 = i[1]
            num3 = i[2]
            num4 = i[3]

        if comb_r == 5:
            num1 = i[0]
            num2 = i[1]
            num3 = i[2]
            num4 = i[3]
            num5 = i[4]



        if num1>0 and num1<= N:
            a[num1-1] = 1
        if num2>0 and num2<= N:
            a[num2-1] = 1
        if num3>0 and num3<= N:
            a[num3-1] = 1

        if num4>0 and num4<= N:
            a[num4-1] = 1
        if num5>0 and num5<=N:
            a[num5-1] = 1
            

        if num1>N and num1<= 2*N+1:
            b[num1-N-1] = 1
        if num2>N and num2<= 2*N+1:
            b[num2-N-1] = 1
        if num3>N and num3<= 2*N+1:
            b[num3-N-1] = 1

        if num4>N and num4<= 2*N+1:
            b[num4-N-1] = 1
        if num5>N and num5<=2*N+1:
            b[num5-N-1] = 1

        if num1>2*N+1 and num1<= 3*N+1:
            c[num1-2*N-2] = 1
        if num2>2*N+1 and num2<= 3*N+1:
            c[num2-2*N-2] = 1
        if num3>2*N+1 and num3<= 3*N+1:
            c[num3-2*N-2] = 1

        if num4>2*N+1 and num4<= 3*N+1:
            c[num4-2*N-2] = 1
        if num5>2*N+1 and num5<=3*N+1:
            c[num5-2*N-2] = 1

        if num1>3*N+1:
            idx_11 = grid_idx_list[num1-3*N-2][0]
            idx_12 = grid_idx_list[num1-3*N-2][1]
            eta[idx_11][idx_12] = k
            eta_affected_1 = True

        if num2 > 3*N+1:
            idx_21 = grid_idx_list[num2-3*N-2][0]
            idx_22 = grid_idx_list[num2-3*N-2][1]
            eta[idx_21][idx_22] = k
            eta_affected_2 = True

        if num3 > 3*N+1:
            idx_31 = grid_idx_list[num3-3*N-2][0]
            idx_32 = grid_idx_list[num3-3*N-2][1]
            eta[idx_31][idx_32] = k
            eta_affected_3 = True


        if num4 > 3*N+1:
            idx_41 = grid_idx_list[num4-3*N-2][0]
            idx_42 = grid_idx_list[num4-3*N-2][1]
            eta[idx_41][idx_42] = k
            eta_affected_4 = True

        if num5 > 3*N+1:
            idx_51 = grid_idx_list[num5-3*N-2][0]
            idx_52 = grid_idx_list[num5-3*N-2][1]
            eta[idx_51][idx_52] = k
            eta_affected_5 = True
        

        alpha = (1-a)*epsilon + a*k
        beta = (1-b)*epsilon + b*k
        gamma = (1-c)*epsilon + c*k        
        X = linear(N,alpha,beta,gamma,eta,mass,L)

        end_eff = ut.end_effector(X,N)
        cm_w_anti,cm_w_clock,start,end = revised_winding.winding(X,N)
        x_info_clock.append(end_eff[0]+0.5)
        y_info_clock.append(end_eff[1]+0.5)
        clock_winding.append(cm_w_clock)

        x_info_anti.append(end_eff[0]-0.5)
        y_info_anti.append(end_eff[1]-0.5)
        anti_winding.append(cm_w_anti)
       

        if count%1000==0:
            print("Count is:",count)

        '''color = check_positions(X,N)
        if color == 1:
            x_val,y_val,x_last,y_last = store_xy(X,len(X))
            x1_val.append(x_val)
            y1_val.append(y_val)
        if color == 2:
            x_val,y_val,x_last,y_last = store_xy(X,len(X))
            x2_val.append(x_val)
            y2_val.append(y_val)
        if color == 3:
            x_val,y_val,x_last,y_last = store_xy(X,len(X))
            x3_val.append(x_val)
            y3_val.append(y_val)
        if color == 4:
            x_val,y_val,x_last,y_last = store_xy(X,len(X))
            x4_val.append(x_val)
            y4_val.append(y_val)
                        
                    
        x_total,y_total,x_last,y_last = store_xy(X,len(X))
        x_info.append(x_total)

        y_info.append(y_total)'''
        

        if eta_affected_1:
            eta[idx_11][idx_12] = epsilon
        if eta_affected_2:
            eta[idx_21][idx_22] = epsilon

        if eta_affected_3:
            eta[idx_31][idx_32] = epsilon

        if eta_affected_4:
            eta[idx_41][idx_42] = epsilon

        if eta_affected_5:
            eta[idx_51][idx_52] = epsilon

        eta_affected_1 = False
        eta_affected_2 = False
        eta_affected_3 = False
        eta_affected_4 = False
        eta_affected_5 = False

        del a,b,c,X
        count += 1

    del comb
    comb_r += 1



eta = make_eta(N,prev_conn,k)

comb_r = 1

while(comb_r<5):
    comb = combinations(comb_n_list,comb_r)

    eta_affected_1 = False
    eta_affected_2 = False
    eta_affected_3 = False
    eta_affected_4 = False
    eta_affected_5 = False

    num1 = 0
    num2 = 0
    num3 = 0
    num4 = 0
    num5 = 0

    for i in list(comb):

        a = np.zeros(N)
        b = np.zeros(N+1)
        c = np.zeros(N)

        if comb_r==1:
            num1 = i[0]

        if comb_r == 2:
            num1 = i[0]
            num2 = i[1]

        if comb_r == 3:
            num1 = i[0]
            num2 = i[1]
            num3 = i[2]

        if comb_r == 4:
            num1 = i[0]
            num2 = i[1]
            num3 = i[2]
            num4 = i[3]

        if comb_r == 5:
            num1 = i[0]
            num2 = i[1]
            num3 = i[2]
            num4 = i[3]
            num5 = i[4]


        if num1>0 and num1<= N:
            a[num1-1] = 1
        if num2>0 and num2<= N:
            a[num2-1] = 1
        if num3>0 and num3<= N:
            a[num3-1] = 1
        if num4>0 and num4<= N:
            a[num4-1] = 1
        if num5>0 and num5<= N:
            a[num5-1] = 1

        if num1>N and num1<= 2*N+1:
            b[num1-N-1] = 1
        if num2>N and num2<= 2*N+1:
            b[num2-N-1] = 1
        if num3>N and num3<= 2*N+1:
            b[num3-N-1] = 1
        if num4>N and num4<= 2*N+1:
            b[num4-N-1] = 1
        if num5>N and num5<= 2*N+1:
            b[num5-N-1] = 1

        if num1>2*N+1 and num1<= 3*N+1:
            c[num1-2*N-2] = 1
        if num2>2*N+1 and num2<= 3*N+1:
            c[num2-2*N-2] = 1
        if num3>2*N+1 and num3<= 3*N+1:
            c[num3-2*N-2] = 1
        if num4>2*N+1 and num4<= 3*N+1:
            c[num4-2*N-2] = 1
        if num5>2*N+1 and num5<= 3*N+1:
            c[num5-2*N-2] = 1

        if num1>3*N+1:
            idx_11 = grid_idx_list[num1-3*N-2][0]
            idx_12 = grid_idx_list[num1-3*N-2][1]
            eta[idx_11][idx_12] = epsilon
            eta_affected_1 = True

        if num2 > 3*N+1:
            idx_21 = grid_idx_list[num2-3*N-2][0]
            idx_22 = grid_idx_list[num2-3*N-2][1]
            eta[idx_21][idx_22] = epsilon
            eta_affected_2 = True

        if num3 > 3*N+1:
            idx_31 = grid_idx_list[num3-3*N-2][0]
            idx_32 = grid_idx_list[num3-3*N-2][1]
            eta[idx_31][idx_32] = epsilon
            eta_affected_3 = True

        if num4 > 3*N+1:
            idx_41 = grid_idx_list[num4-3*N-2][0]
            idx_42 = grid_idx_list[num4-3*N-2][1]
            eta[idx_41][idx_42] = epsilon
            eta_affected_4 = True

        if num5 > 3*N+1:
            idx_51 = grid_idx_list[num5-3*N-2][0]
            idx_52 = grid_idx_list[num5-3*N-2][1]
            eta[idx_51][idx_52] = epsilon
            eta_affected_5 = True

        alpha = a*epsilon + (1-a)*k
        beta = b*epsilon + (1-b)*k
        gamma = c*epsilon + (1-c)*k        
        X = linear(N,alpha,beta,gamma,eta,mass,L)

        '''cm_w,start,end = revised_winding.winding(X,N)
        stiffness = effective_stiffness(start,end,alpha)

        if cm_w>300 or stiffness>10:
            #To display the structure of the trunk
            if cm_w>300:
                show_trunk.save_trunk_unstretched(X,len(X),L,'_w_',cm_w)
                print("Winding Number:",cm_w)
                print("Values of alpha: ",alpha)
                print("Values of beta: ",beta)
                print("Values of gamma: ",gamma)
                print("Values of eta:",eta)
                print("Position:",X)
            if stiffness>10:
                show_trunk.save_trunk_unstretched(X,len(X),L,'_s_',stiffness)
                print("Stiffness Value",stiffness)
                print("Values of alpha: ",alpha)
                print("Values of beta: ",beta)
                print("Values of gamma: ",gamma)
                print("Values of eta:",eta)
                print("Position:",X)

    
        winding.append(cm_w)
        stiff.append(stiffness)'''

        if count%1000==0:
            print("Count is:",count)

        '''color = check_positions(X,N)
        if color == 1:
            x_val,y_val,x_last,y_last = store_xy(X,len(X))
            x1_val.append(x_val)
            y1_val.append(y_val)
        if color == 2:
            x_val,y_val,x_last,y_last = store_xy(X,len(X))
            x2_val.append(x_val)
            y2_val.append(y_val)
        if color == 3:
            x_val,y_val,x_last,y_last = store_xy(X,len(X))
            x3_val.append(x_val)
            y3_val.append(y_val)
        if color == 4:
            x_val,y_val,x_last,y_last = store_xy(X,len(X))
            x4_val.append(x_val)
            y4_val.append(y_val)
                        
                    
        x_total,y_total,x_last,y_last = store_xy(X,len(X))
        x_info.append(x_total)

        y_info.append(y_total)'''

        end_eff = ut.end_effector(X,N)
        cm_w_anti,cm_w_clock,start,end = revised_winding.winding(X,N)
        x_info_clock.append(end_eff[0]+0.5)
        y_info_clock.append(end_eff[1]+0.5)
        clock_winding.append(cm_w_clock)

        x_info_anti.append(end_eff[0]-0.5)
        y_info_anti.append(end_eff[1]-0.5)
        anti_winding.append(cm_w_anti)
        

        if eta_affected_1:
            eta[idx_11][idx_12] = k
        if eta_affected_2:
            eta[idx_21][idx_22] = k

        if eta_affected_3:
            eta[idx_31][idx_32] = k

        if eta_affected_4:
            eta[idx_41][idx_42] = k

        if eta_affected_5:
            eta[idx_51][idx_52] = k


        eta_affected_1 = False
        eta_affected_2 = False
        eta_affected_3 = False
        eta_affected_4 = False
        eta_affected_5 = False

        del a,b,c,X
        count += 1

    del comb
    comb_r += 1
                


a = np.zeros(N)
b = np.zeros(N+1)
c = np.zeros(N)

alpha = a*epsilon + (1-a)*k
beta = b*epsilon + (1-b)*k
gamma = c*epsilon + (1-c)*k        
X = linear(N,alpha,beta,gamma,eta,mass,L)

end_eff = ut.end_effector(X,N)
cm_w_anti,cm_w_clock,start,end = revised_winding.winding(X,N)
x_info_clock.append(end_eff[0]+0.5)
y_info_clock.append(end_eff[1]+0.5)
clock_winding.append(cm_w_clock)

x_info_anti.append(end_eff[0]-0.5)
y_info_anti.append(end_eff[1]-0.5)
anti_winding.append(cm_w_anti)

if count%1000==0:
    print("Count is:",count)


print("total N",comb_n, "eta n:",tot_eta)
#print("list:",comb_n_list)

#To set the lowest values in the scatter
x_info_clock.append(25)
y_info_clock.append(25)
clock_winding.append(0)

x_info_anti.append(25)
y_info_anti.append(25)
anti_winding.append(0)


plt.scatter(x_info_clock,y_info_clock,c = clock_winding,cmap=plt.cm.Blues, s = 4)
plt.scatter(x_info_anti,y_info_anti,c = anti_winding,cmap=plt.cm.Reds, s = 4)
plt.gca().invert_yaxis()
plt.show()

#Used to show the direction from which the end point is reached
#show_trunk.show_grid_end_effector_orientation(x_info,y_info,L,loops)

#Used to show the various end points reached
#show_trunk.show_end_point(x1_val,y1_val,x2_val,y2_val,x3_val,y3_val,x4_val,y4_val,L)

#Used to show the mean and standard deviation of the various end points reached
#grid_show(np.asarray(x_info),np.asarray(y_info),L)


#revised_winding.winding_plot(np.asarray(winding),np.asarray(stiff),2)
#print(winding)
#print(stiff)
    
