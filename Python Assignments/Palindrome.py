x = input("Enter word to check palindrome or not: ")

res = str(x) == str(x)[::-1]

print("Is the word palindrom? " + str(res))
