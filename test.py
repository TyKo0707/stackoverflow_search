import random

paths = []

# Check if in my turn (l == 1 and r == 1) or (r == 1) or (l == 1)
for i in range(3):
    if i == 0:
        l = 4
        r = 4
    elif i == 1:
        l = 5
        r = 4
    else:
        l = 4
        r = 5
    path = ''
    while not ((l == 1 and r == 1) or (l == 1 and r == 0) or (l == 0 and r == 1)):
        while r >= 0 and l >= 0:
            ran = random.randint(1, 3)
            if ran == 1:
                l -= 1
                path += 'l-1\t'
            elif ran == 2:
                l -= 1
                r -= 1
                path += 'b-1\t'
            else:
                r -= 1
                path += 'r-1\t'
    else:
        paths.append(path)

print(paths)
