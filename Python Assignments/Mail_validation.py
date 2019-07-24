import string as st

print('Mail Validation\n\n')
mail=input("Enter Mail ID to Validate: ")

def chk_mail():
    try:
        
        first_char = mail[0]
        sym_mail = mail.find('@') # will give object index
        char_chk1 = mail[mail.find('@')+1:mail.find('.')]
        char_chk2 = mail.split('.')
        #print(char_chk1)
        #print(char_chk2)

        if mail[0] in st.ascii_letters:
            if sym_mail > 0:
                if len(char_chk1) > 3 and type(char_chk1)==str:
                    if len(char_chk2[1]) <= 3 and type(char_chk2[1])==str:
                            print("Valid Email")
                    else:
                        print("'.' followed by Domain name. Make sure character is 3 letter")
                else:
                        print("'@' should be followed by characters(say gmail not digits) till '.'")
            else:
                print("Special Character (@) are not founded")
        else:
            print("Make sure Mail ID should start with Alphabet")
        
    except ValueError:
        print('Not a valid format')


chk_mail()
