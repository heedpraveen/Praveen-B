'''
Program1: Get the string from user and print the character one by one.

Program2: Calculate length of the string without any inbuilt function. Try to calculate length using loop.
'''

user = input("Enter String: ")
count =0
for i in user:
    print(i) # Program1
    count += 1

print('\n',count) # Pragram2

'''
Program3: Get two strings from the user and create a new string by swapping the first
characters in each of the string.

For ex: input -->'abc' 'xyz'
        output --> xbc  ayz 

'''
text_1 = input('First string: ')
text_2 = input('Second string: ')

interchange_first_letter_1 = text_1[:1] + text_2[1:]
interchange_first_letter_2 = text_2[:1] + text_1[1:]

print(interchange_first_letter_1,'\t',interchange_first_letter_2)

'''
Program4: Get a list of string from user and print longest word along with its length
'''
text1 = input('String: ')
list_user = text1.split()
#print(list_user)
max_string = max(len(x) for x in list_user)
string = max(list_user, key=len)
print(string,max_string)


'''
Program5: Write a python function to convert a given string to all uppercase if it contains
atleast 2 uppercase characters in the first 4 characters. Else convert it to lowercase
'''
n = input('>>')

count =0
for x in n[:5]:
	if x.isupper():
		count += 1

print(count) #Upper case count
if count >= 2:
    print(n.upper())
else:
    print(n.lower())


    

