'''
The methods of the class engine in module inflect.py provide plural
inflections, singular noun inflections, “a”/”an” selection for English
words, and manipulation of numbers as words.

Question: Print word of a given number

    EG:  INPUT --> 98
         OUTPUT --> ninety-eight

Before importing below library make sure it is installed. If not use 'pip install inflect,num2words'
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
