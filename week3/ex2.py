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
x_bar=sum(N)/n
y_bar=sum(M)/n

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
print("The covariance betweeen two vectors : ")
cov_x_y=print(sumation(N,M,n,x_bar,y_bar)/n-1)




