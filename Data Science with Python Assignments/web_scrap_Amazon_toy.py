import requests
from bs4 import BeautifulSoup
import smtplib


def check_price():
    URL = "https://www.amazon.in/Giant-Innovative-Stylish-Sunglasses-Aviator/dp/B07MPSZ7NK/ref=bbp_bb_d33a38_st_CCqh_w_0?psc=1&smid=ANTJPL45A9VCN"

    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'}

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

    server.login('praveenciet2011@gmail.com', 'Wiprotec@123')

    sub = 'Alert! price reduced'
    body = 'Click on the following amazon link: https://www.amazon.in/Giant-Innovative-Stylish-Sunglasses-Aviator/dp/B07MPSZ7NK/ref=bbp_bb_d33a38_st_CCqh_w_0?psc=1&smid=ANTJPL45A9VCN'
    msg =f'Subject: {sub}\n\n{body}'

    server.sendmail(
        'praveenciet2011@gmail.com',
        'praveenb9629@gmail.com',
        msg)

    print("Hey mail has been sent")

    server.quit()
    

    

    
check_price()    
