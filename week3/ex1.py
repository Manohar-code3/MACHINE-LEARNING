#create 1xN array
n=int(input("enter the n (no of elements) = "))
alist=[]
for i in range(n):
    x=int(input("enter the element = "))
    alist.append(x)
sum=0
for i in alist:
    sum+=i
mean=sum/n
def sam_var(alist,n,mean):
    blist=[]
    for i in alist:
        s1=pow(i-mean,2)
        blist.append(s1)
    sum=0
    for j in blist:
        sum+=j
    sv=(sum)/n-1
    return sv
x=sam_var(alist,n,mean)
print("sample varaince = ",x)
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
y=pop_var(alist,n,mean)
print("pouplation variance = ",y)
# print("mean = ",mean)




