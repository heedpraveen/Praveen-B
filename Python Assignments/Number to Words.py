'''
The methods of the class engine in module inflect.py provide plural
inflections, singular noun inflections, “a”/”an” selection for English
words, and manipulation of numbers as words.

Here have used number_to_words function only
'''

import inflect

x = inflect.engine()
user=input('Enter number: ')
print(x.number_to_words(user))



'''
Another library module we can user is num2words
'''

import num2words
user=input('Enter number: ')
print(num2words.num2words(user))
