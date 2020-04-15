a = int(input()); time = {}; old = 2**31; ans= 1
for i in range(a):
    k,j = map(int,input().split())
    if(k in time):
        if(j-k<time[k]-k):
            time[k]=j
    else :
        time[k]=j

time = sorted(time.items())
index=0
for i in time:
    if(old>i[1]):
        old = i[1]
        index =i
for i in time:
    if(i==index and i[0]==i[1]):
        ans -=1
    if(old<=i[0]):
        old = i[1]
        ans +=1
print(ans)