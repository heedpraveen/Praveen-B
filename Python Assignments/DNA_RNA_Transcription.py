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

