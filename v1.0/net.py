import requests
import muggle_ocr
import os
import denoise_pict


cookie_session_id = 'F9CBE6CFC7CA699DDB8671D5B568A16D'
root_dir = 'E:/Trash/AutoRancode/dataset/train/'


def ocr():
    denoise_pict.pro_img(r"./ocr.png", r"./ocr_temp.png")
    with open(r"./ocr_temp.png", "rb") as f:
        b = f.read()
    text = sdk.predict(image_bytes=b)
    f.close()
    return text


def check(rancode):
    url = 'https://sso.scnu.edu.cn/AccountService/user/checkrandom.html'
    headers = {
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'cookie': 'cookie_session_id=' + cookie_session_id
    }
    params = {"random": rancode}
    response = requests.post(url, data=params, headers=headers)
    print(response.json())
    if response.json()['msgcode'] == 0:
        return True
    else:
        return False


def getimg():
    url = "https://sso.scnu.edu.cn/AccountService/user/rancode.jpg"
    headers = {
        'accept': 'image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'cookie': 'cookie_session_id=' + cookie_session_id
    }
    res = requests.get(url=url, headers=headers).content
    with open("./ocr.png", "wb") as f:
        f.write(res)
    f.close()


def getimgs(n):
    num = 0
    total = 0
    while(num < n):
        getimg()
        total += 1
        rancode = ocr()
        if check(rancode):
            try:
                os.rename('ocr.png', rancode+'.png')
                num += 1
                print('{} / {}'.format(num, n))
            except:
                pass
    return n/total


if __name__ == '__main__':
    sdk = muggle_ocr.SDK(model_type=muggle_ocr.ModelType.OCR)
    os.chdir(root_dir)
    rate = getimgs(5000)
    print('准确率 {}%'.format(rate*100))
 