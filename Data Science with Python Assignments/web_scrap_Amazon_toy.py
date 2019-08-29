import requests
from bs4 import BeautifulSoup
import smtplib


def check_price():
    URL = "https://www.amazon.in/Giant-Innovative-Stylish-Sunglasses-Aviator/dp/B07MPSZ7NK/ref=bbp_bb_d33a38_st_CCqh_w_0?psc=1&smid=ANTJPL45A9VCN"

    headers = "type in google to get your user agent(Like, my user agent) and write as {'User_Agent':'copy the whole link displayed in google search'}"

    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    title = soup.find(id="productTitle").get_text()
    price = soup.find(id="priceblock_ourprice").get_text()
    converted_price = float(price[1:5])

    if converted_price > 150.0:
        send_mail()
    
    print(title.strip())
    print(price)
    print(converted_price)

def send_mail():
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()

    server.login('your mail-id', 'your-password')

    sub = 'Alert! price reduced'
    body = 'Click on the following amazon link: https://www.amazon.in/Giant-Innovative-Stylish-Sunglasses-Aviator/dp/B07MPSZ7NK/ref=bbp_bb_d33a38_st_CCqh_w_0?psc=1&smid=ANTJPL45A9VCN'
    msg =f'Subject: {sub}\n\n{body}'

    server.sendmail(
        'from(your mail-id)',
        'to recepient mail-id',
        msg)

    print("Hey mail has been sent")

    server.quit()
    

    

    
check_price()    
