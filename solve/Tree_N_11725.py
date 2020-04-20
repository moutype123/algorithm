node_Number = int(input()); tree = {}
check = {1:1}
for i in range(1,node_Number):
    key, value = map(int,input().split())
    if value in check:
        tree[key] = value
        check[key] = key
        continue
    tree[value] = key

for i in range(2,node_Number+1):
    print(tree[i])

