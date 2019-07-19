import requests
from bs4 import BeautifulSoup
import smtplib

 
def check_price():
    URL = "https://www.amazon.in/15-6-inch-FireCuda-Windows-Graphics-FX504GM-EN394T/dp/B07LDJ7ZHW/ref=sr_1_16?crid=1CP548XBFIU2F&keywords=asus+rog+laptops&qid=1563540270&s=gateway&sprefix=asus+rog%2Caps%2C295&sr=8-16"

    headers = {'User_Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'}

    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    title = soup.find(id='productTitle').get_text()
    price = soup.find(id='priceblock_ourprice').get_text()
    price_var = price.replace(',','')
    conv_price = price_var[2:11]
      
    conv_price_type = float(conv_price)

    if conv_price_type > 100000.0: # Change depending on your need
        send_mail()
    
    print(title.strip())
    print(price)


def send_mail():
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()

    server.login('Mail ID','Your-Password')

    sub = '!!!Gaming Laptop - ASUS ROG!!! Price reduced'
    body = 'This is just an unbelievable price cut.\nCheck the following link: https://www.amazon.in/15-6-inch-FireCuda-Windows-Graphics-FX504GM-EN394T/dp/B07LDJ7ZHW/ref=sr_1_16?crid=1CP548XBFIU2F&keywords=asus+rog+laptops&qid=1563540270&s=gateway&sprefix=asus+rog%2Caps%2C295&sr=8-16'
    msg = f'Subject: {sub}\n\n{body}'

    server.sendmail(
        'sender',
        ['recipient'],
        msg
        )

    print('The price drop has been intimidated to the mail recipients')

    server.quit()
    
check_price()    
