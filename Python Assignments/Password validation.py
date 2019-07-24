'''
Password Policy:

Length greater than or equal to 8
Atleast one Upper letter
Atleast one lower letter
Atleast one digits
Atleast one special character
'''
def passwd_policy():
    print("Password Validity\n\n")
    print("Enter your password below to check with policy\n")
    user = input(">")
    symbol = ['?','&','^','%','#','$','@','!']
    if len(user) >= 8:
        if any(x.isupper() for x in user):
            if any(x.islower() for x in user):
                if any(x.isdigit() for x in user):
                    for x in symbol:
                        if x in user:
                            print("Password looks Strong")
                            break
                    else:
                        print("Password must have atlest a symbol in it")
                else:
                    print('As per policy password must have atlest a digit in it')
            else:
                print("password must to have lower letter in it")
        else:
            print('must to have upper letter in it')

    else:
        print("bad password length")
                    

passwd_policy()
