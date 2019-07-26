'''
Write the program such that there would not be any number in the code.
Ouput: 420
'''
print(ord('d')+ord('d')+ord('d')+ord('x'))

'''
You will get Unicode integer of given string

FYI: Unicode value will get vary with respect to uppercase, lowercase and punctuation


import string
[ord(x) for x in string.ascii_lowercase]
[97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

[ord(x) for x in string.ascii_uppercase]
[65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]

[ord(x) for x in string.punctuation]
[33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 58, 59, 60, 61, 62, 63, 64, 91, 92, 93, 94, 95, 96, 123, 124, 125, 126]
'''

# or

print(sum(ord(x) for x in 'Jado&'))
