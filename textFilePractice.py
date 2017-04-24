
file = open('README.md','r')

lines = file.readlines()
print(lines)

for line1 in lines:
    print(line1.strip())