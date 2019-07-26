import time
import math

def prime_num_v1(n):#Method 1
    if n==1:
        return False

    for d in range(2,n):
        if n%d==0:
            return False
    return True


def prime_num_v2(n): #Method 2
    if n==1:
        return False

    max_divisor = math.floor(math.sqrt(n))
    for d in range(2, 1+max_divisor):
        if n%d==0:
            return False
    return True    


n = int(input("Enter number: "))
print('Is this a prime number? ',n, prime_num_v1(n))
'''      
To check time of process
t0 = time.time()
for n in range(1, 100):
    print(n, prime_num_v2(n))
t1=time.time()
print('Time required: ', t1-t0)
'''
