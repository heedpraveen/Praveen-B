'''
HACKEREARTH CHALLENGE:

Jadoo, the Space Alien has befriended Koba upon landing on Earth. Since then, he wishes Koba to be more like him. 
In order to do so he decides to slowly transcribe Koba's DNA into RNA. But he has to write a very short code in order
to do the transcription so as not to make Koba aware of the change.

The four nucleotides found in DNA are adenine (A), cytosine (C), guanine (G) and thymine (T).

The four nucleotides found in RNA are adenine (A), cytosine (C), guanine (G) and uracil (U).

Given a DNA strand, its transcribed RNA strand is formed by replacing each nucleotide with its complement:

G --> C
C --> G
T --> A
A --> U
 

Input: The input will always be a string of characters.

Output: The output should always be a string of characters. In the case of invalid input, you should output Invalid Input as a string.

            Rules: Your code should not consist of any numerical characters             
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) and the length of your code should be <= 103.
            If your code consists of numerical characters, then your score is zero irrespectuve of your code length or testcases 
            satisfied. If your code is devoid of numerical characters and is of length > 103 then you score 50%.
'''

import string
dna = input()
dna_strand = ['G','C','T','A']
trans_table = str.maketrans('GCTA','CGAU')
rna = str.translate(dna,trans_table)
for x in dna:
    if x not in dna_strand:
        print('Invalid Input')
        break
    else:
        pass
else:
    print(rna)

