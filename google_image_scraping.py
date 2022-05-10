from fileinput import filename
from bs4 import BeautifulSoup
import requests
import shutil,os


def scrape_imgs_links(url):
    req = requests.get(url)
    #print(req.content)

    soup = BeautifulSoup(req.content, "lxml")
    #print(soup.prettify())
    body = soup.find('body')
    tags = body.find_all('img')

    imgs_link=[]
    for i, tag in enumerate(tags):
        if i!=0:
            #print(tag['src'])
            imgs_link.append(tag['src'])
    #print(imgs_link)
    return imgs_link



def download(image_url, filename):
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream = True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
            
        print('Image sucessfully Downloaded: ',filename)
    else:
        print('Image Couldn\'t be retreived')



def main():
    """ urls = ['https://www.google.com/search?q=a+men+carrying+hammer+2&tbm=isch&ved=2ahUKEwil6Jug3LP2AhVU_TgGHf-GDkoQ2-cCegQIABAA&oq=a+men+carrying+hammer+2&gs_lcp=CgNpbWcQA1C8AVi8AWDTBmgAcAB4AIABxQGIAYoDkgEDMC4ymAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=6dQlYuXPGtT64-EP_4260AQ&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983',
            'https://www.google.com/search?q=two+men+carrying+knife&tbm=isch&ved=2ahUKEwiJ_OzH3LP2AhXV_DgGHRNrCMYQ2-cCegQIABAA&oq=two+men+carrying+knife&gs_lcp=CgNpbWcQA1AAWPAMYPMVaAFwAHgAgAHfAYgBnwmSAQUwLjUuMZgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=PNUlYonuIdX54-EPk9ahsAw&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983',
            'https://www.google.com/search?q=a+men+carrying+knife&tbm=isch&ved=2ahUKEwjl1t_P3LP2AhXWzqACHaQmAyIQ2-cCegQIABAA&oq=a+men+carrying+knife&gs_lcp=CgNpbWcQAzoHCCMQ7wMQJ1CfHlifHmD0IWgAcAB4AIAB_gGIAb4DkgEFMC4xLjGYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=TdUlYqX8Btadg8UPpM2MkAI&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983',
            'https://www.google.com/search?q=boy+carrying+knife&tbm=isch&ved=2ahUKEwjp97jl3LP2AhXTi9gFHUnZAxEQ2-cCegQIABAA&oq=boy+carrying+knife&gs_lcp=CgNpbWcQAzoICAAQCBAHEB5Q7AlY5Q5gixRoAHAAeACAAb4BiAHqBZIBAzAuNJgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=etUlYunSJdOX4t4PybKPiAE&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983',
            'https://www.google.com/search?q=teenager+carrying+knife&tbm=isch&ved=2ahUKEwjN3I7w3LP2AhUyUXwKHZqyDQEQ2-cCegQIABAA&oq=teenager+carrying+knife&gs_lcp=CgNpbWcQA1CVFljdImDuLGgAcAB4AIABxgGIAcMNkgEDMC45mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=kNUlYs3UPLKi8QOa5bYI&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983',
            'https://www.google.com/search?q=knife+in+human+hand&rlz=1C1MSIM_enNP983NP983&hl=en&sxsrf=APq-WBsSQPCphC9W86CppIiH3IJi0CAWhA:1646646763669&source=lnms&tbm=isch&sa=X&ved=2ahUKEwj_paSb3bP2AhUWCd4KHbbSBL8Q_AUoAXoECAEQAw&biw=1920&bih=969&dpr=1',
            'https://www.google.com/search?q=knives+in+human+hand&tbm=isch&ved=2ahUKEwiqrsKd3bP2AhW2yKACHY4DC8UQ2-cCegQIABAA&oq=knives+in+human+hand&gs_lcp=CgNpbWcQA1DoDFjVImCnJGgBcAB4AIAB3gGIAcMKkgEFMC42LjGYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=8NUlYqr2DLaRg8UPjoesqAw&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=knives+pointing+to+human+&tbm=isch&ved=2ahUKEwjyw_Go3bP2AhWp3nMBHQhGC74Q2-cCegQIABAA&oq=knives+pointing+to+human+&gs_lcp=CgNpbWcQAzoFCAAQgAQ6BggAEAgQHlDvClihKmDJLGgAcAB4AYABiQKIAcAdkgEGMC4xNS40mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=CNYlYvKfA6m9z7sPiIyt8As&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=knives+pointing+to+girl&tbm=isch&ved=2ahUKEwjX2pK33bP2AhUsYmwGHXwXDjEQ2-cCegQIABAA&oq=knives+pointing+to+girl&gs_lcp=CgNpbWcQA1AAWLkEYKkIaABwAHgAgAHCAYgB8wWSAQMwLjSYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=JdYlYpe0OqzEseMP_K64iAM&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=gun+in+hand&tbm=isch&ved=2ahUKEwiLzJvX3bP2AhUMgWMGHU2UCKkQ2-cCegQIABAA&oq=gun+in+hand&gs_lcp=CgNpbWcQAzIECAAQQzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDoGCAAQBRAeOgYIABAIEB46BAgAEBg6CAgAEIAEELEDOgsIABCABBCxAxCDAToHCAAQsQMQQ1DhDljSImDfI2gAcAB4AIAB3QGIAd8QkgEFMC45LjKYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=adYlYsvzDIyCjuMPzaiiyAo&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=gun+in+human+hand&tbm=isch&ved=2ahUKEwjapuDd3bP2AhWWR2wGHcOSDD8Q2-cCegQIABAA&oq=gun+in+human+hand&gs_lcp=CgNpbWcQAzoECAAQQzoFCAAQgARQ0wlY0xBgwRNoAHAAeACAAfIBiAHZCpIBBTAuNi4xmAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=dtYlYtqTOJaPseMPw6Wy-AM&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=gun+pointing+at+camera+&tbm=isch&ved=2ahUKEwijyOb43bP2AhWfktgFHZfMAfEQ2-cCegQIABAA&oq=gun+pointing+at+camera+&gs_lcp=CgNpbWcQAzIECAAQQzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABFCjCFjzDmCrEWgAcAB4AIABxgGIAYoJkgEDMC42mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=r9YlYuO0J5-l4t4Pl5mHiA8&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=gun+pointing+at+other&tbm=isch&ved=2ahUKEwjT7OP63bP2AhW3gGMGHQXDBCoQ2-cCegQIABAA&oq=gun+pointing+at+other&gs_lcp=CgNpbWcQAzIGCAAQCBAeMgYIABAIEB5QAFiSBWC9B2gAcAB4AIABsgKIAb8IkgEHMC4zLjEuMZgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=s9YlYpPHMLeBjuMPhYaT0AI&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=gun+pointing+at+other+person&tbm=isch&ved=2ahUKEwiYi_nu4LP2AhULXWwGHQxbDA0Q2-cCegQIABAA&oq=gun+pointing+at+other+person&gs_lcp=CgNpbWcQAzoGCAAQCBAeOgcIIxDvAxAnUI0aWOlcYPtfaA1wAHgAgAHQAYgB0BySAQYwLjE2LjOYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=wNklYpirEYu6seMPjLaxaA&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=gun+pointing+at+crowd&tbm=isch&ved=2ahUKEwi0tdWF4bP2AhVZ9jgGHf4gBQwQ2-cCegQIABAA&oq=gun+pointing+at+crowd&gs_lcp=CgNpbWcQAzoHCCMQ7wMQJzoGCAAQCBAeOgUIABCABFCEFljcQGCyQmgBcAB4AIAB2QGIAe8ckgEGMC4xNS40mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=79klYvSBOdns4-EP_sGUYA&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=gun+pointing+at+room&tbm=isch&ved=2ahUKEwiA6Omr47P2AhVIR2wGHeGqAPgQ2-cCegQIABAA&oq=gun+pointing+at+room&gs_lcp=CgNpbWcQAzoHCCMQ7wMQJzoFCAAQgAQ6BggAEAgQHjoECAAQGFCbEljzFmCCGmgAcAB4AIABxAGIAcAHkgEDMC41mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=WNwlYoDcMsiOseMP4dWCwA8&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=people+with+robbery+mask&tbm=isch&ved=2ahUKEwj3h_b647P2AhXhzqACHU4bCAsQ2-cCegQIABAA&oq=people+with+robbery+mask&gs_lcp=CgNpbWcQAzoECAAQQzoFCAAQgAQ6BggAEAcQHjoICAAQCBAHEB5Q_QlYyiRgzSdoAHAAeACAAc8BiAHOE5IBBjAuMTEuMpgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=_twlYveQK-Gdg8UPzragWA&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=group+of+people+with+robbery+mask&tbm=isch&ved=2ahUKEwjLlMGD5LP2AhVQW2wGHYekC3EQ2-cCegQIABAA&oq=group+of+people+with+robbery+mask&gs_lcp=CgNpbWcQA1CVB1jyNGDrO2gEcAB4AIABzAGIAZMVkgEGMC4xMC40mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=EN0lYsvMK9C2seMPh8muiAc&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=+robbery+mask+on+boys+face&tbm=isch&ved=2ahUKEwiy6b2e5LP2AhWGLrcAHU0XDsMQ2-cCegQIABAA&oq=+robbery+mask+on+boys+face&gs_lcp=CgNpbWcQAzoHCCMQ7wMQJzoFCAAQgAQ6BAgAEB46BggAEAgQHlDVLFipTWDVT2gAcAB4AIABzAGIAc0WkgEGMC4xMy4ymAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=Sd0lYvKgEYbd3LUPza64mAw&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=thiefing+mask&tbm=isch&ved=2ahUKEwjBh_2z5LP2AhV0SHwKHdHUD8oQ2-cCegQIABAA&oq=thiefing+mask&gs_lcp=CgNpbWcQAzoECAAQQzoFCAAQgAQ6BwgjEO8DECc6CAgAEIAEELEDOgsIABCABBCxAxCDAToGCAAQBRAeOgQIABAYOgYIABAKEBhQ8BZYtEVg_EloAnAAeACAAdwBiAHhFZIBBTAuNy43mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=dt0lYsH0FfSQ8QPRqb_QDA&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=robbery+mask+on+face+with+gun+in+hand&tbm=isch&ved=2ahUKEwj5w-_N5LP2AhXXk9gFHThWDqgQ2-cCegQIABAA&oq=robbery+mask+on+face+with+gun+in+hand&gs_lcp=CgNpbWcQAzoHCCMQ7wMQJ1ClA1igLWD2LmgBcAB4AIABzQGIAdsckgEGMC4xNC41mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=rN0lYvm9KNen4t4PuKy5wAo&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=robbery+mask+on+face+with+gun+in+2+hand&tbm=isch&ved=2ahUKEwiYnKjR5LP2AhWyRGwGHazrCnAQ2-cCegQIABAA&oq=robbery+mask+on+face+with+gun+in+2+hand&gs_lcp=CgNpbWcQA1DsC1iQDmD_FWgAcAB4AIAByAGIAckEkgEFMC4yLjGYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=s90lYtj2NbKJseMPrNergAc&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=robbery+mask+on+face+with+knife+in+2+hand&tbm=isch&ved=2ahUKEwjRr93b5LP2AhVB6DgGHQugDgUQ2-cCegQIABAA&oq=robbery+mask+on+face+with+knife+in+2+hand&gs_lcp=CgNpbWcQA1CtCFjbFWCOF2gAcAB4AIABxgGIAf0IkgEDMC42mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=yd0lYpGnLMHQ4-EPi8C6KA&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en',
            'https://www.google.com/search?q=robbery+mask+on+face+with+knife+in+hand&tbm=isch&ved=2ahUKEwjK26Pl5LP2AhXzjNgFHVrjCBIQ2-cCegQIABAA&oq=robbery+mask+on+face+with+knife+in+hand&gs_lcp=CgNpbWcQA1DMCFieDWCrEGgAcAB4AIABywGIAcwEkgEFMC4xLjKYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=3d0lYor5LfOZ4t4P2sajkAE&bih=969&biw=1920&rlz=1C1MSIM_enNP983NP983&hl=en'
            ] """
    urls = []
    count=581
    for url in urls:
        imgs_link = scrape_imgs_links(url)
        
        for img_link in imgs_link:
            if not os.path.exists('./images'):
                os.mkdir('./images')
            filename = './images/' + str(count) + '.jpg'
            download(img_link, filename)
            count += 1


if __name__=="__main__":
    main()