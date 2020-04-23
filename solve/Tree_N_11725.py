import sys
n = int(input());t = {i:[] for i in range(1,n+1) };s = [1];p = {}
for _ in range(n-1):
    a,b = map(int, sys.stdin.readline().split())
    t[a].append(b)
    t[b].append(a)
while s:
    a = s.pop(0)
    for i in t[a]:
        s.append(i)
        p[i] = a
        t[i].remove(a)
for i in range(2,n+1):
    print(p[i])