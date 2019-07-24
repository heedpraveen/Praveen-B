PROJECT TITLE
    
    Web scrapping - Amazon Product

PRE-REQUISITES

    1. Python IDE(as you prefer)
      
      Module
      
    2. request
    3. bs4
    4. smtplib
    
INSTALLATION

    Kindly follow the below steps to install module(Hope you installed Python IDE in your system)
    
    For installing modules in windows,
    
      Run below command in 'cmd' with admin privilege
      
      * pip install requests
      * pip install bs4
      * pip install smtplib
      
    Make use of the following link to get pip instalation code: https://pypi.org/      

DEPLOYMENT PROCEDURE
    
    import required modules 
    Take random product from Amazon and assign it to the variable
    #headers is an optional, here we paste User-agent. This link is for reference about user agent https://www.whoishostingthis.com/tools/user-agent/
    Using requests module get all kind of HTTP request
    parser the HTML page to access 
    Click inspect on Product Title and Price
    find the 'id' of title and price and print the text
    set a condition
    using smtplib module get access to gmail server
    use starttls()[For secure connection] as a sandwich between ehlo() 
    command to login to your mail(make sure your mail has enabled LessSecureConnection)
    type subject, body and message(combine subject and body)
    type command for send mail in the format of {From,To,Message}
    quit from server
     
