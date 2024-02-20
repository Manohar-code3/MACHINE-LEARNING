N=[]
M=[]
n=int(input("enter the n = "))
# m=int(input("enter the m = "))
print("enter the X values")
for i in range(n):
    x=int(input(f"enter the (0,{i}) = "))
    N.append(x)
print("enter the Y values")
for j in range(n):
    y=int(input(f"enter the (0,{j}) = "))
    M.append(y)
x_bar=sum(N)
y_bar=sum(M)

def sumation(arr_of_x,arr_of_y,n,x_bar,y_bar):

    X=[]
    Y=[]
    Z=[]
    for i  in range(n):
        x_div=arr_of_x[i]-x_bar
        X.append(x_div)
        y_div=arr_of_y[i]-y_bar
        Y.append(y_div)
    for j in range(n):
        mul=X[j]*Y[j]
        Z.append(mul)
    return sum(Z)
def pop_var(alist,n,mean):
    blist=[]
    for i in alist:
        s1=pow(i-mean,2)
        blist.append(s1)
    sum=0
    for j in blist:
        sum+=j
    pv=pow(((sum)/n),0.5)
    return pv

print("The covariance betweeen two vectors : ")
cov_x_y=sumation(N,M,n,x_bar,y_bar)/n-1
print(cov_x_y)
print("the var of x : ")
var_x=pop_var(N,n,x_bar)
print(var_x)
print("the var of y : ")
var_y=pop_var(M,n,y_bar)
print(var_y)
print("The correaltion between two vector : ")
correaltion=cov_x_y/(var_x*var_y)
print(correaltion)



