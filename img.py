from time import sleep
import requests
import base64
import os
import re


def getimg(n):
    url = "https://sso.scnu.edu.cn/AccountService/user/rancode.jpg"
    for i in range(n):
        res = requests.get(url=url).content
        with open("./dataset/train/ocr"+str(i)+".png", "wb") as f:
            f.write(res)
        print('{} / {}'.format(i+1, n))
        f.close()


def gettoken():
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=WnobeSMIMF7taYISVMwWyIU0&client_secret=nficnnW7Fex45We750OjWMcIi5nOjpMu'
    response = requests.get(host)
    if response:
        print(response.json())


def ocr():
    token = '24.5699201f0c03e4e807c2301baa6421ff.2592000.1664853812.282335-27294082'
    # request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    request_url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic'
    root_dir = 'E:/Trash/AutoRancode/dataset/train'
    os.chdir(root_dir)
    for image_name in os.listdir(root_dir):
        if 'ocr' in image_name:
            f = open(root_dir+'/'+image_name, 'rb')
            img = base64.b64encode(f.read())
            params = {"image": img}
            request_url = request_url + "?access_token=" + token
            headers = {'content-type': 'application/x-www-form-urlencoded'}
            response = requests.post(request_url, data=params, headers=headers)
            f.close()
            if response:
                if 'error_code' in response.json():
                    os.remove(image_name)
                    print('{} error'.format(image_name))
                else:
                    res = ''
                    if len(response.json()['words_result']) > 0:
                        res = str(response.json()[
                                  'words_result'][0]['words']).replace(' ', '')
                    if len(res) == 4 and res.isalnum():
                        try:
                            os.rename(image_name, res+'.png')
                            print('{} -> {}'.format(image_name, res))
                        except:
                            os.remove(image_name)
                            print('{} remove'.format(image_name))
                    else:
                        os.remove(image_name)
                        print('{} remove'.format(image_name))
            sleep(0.5)


def check():
    pattern = re.compile(r'[a-zA-Z0-9]{4}')
    root_dir = 'E:/Trash/AutoRancode/dataset/train'
    os.chdir(root_dir)
    for image_name in os.listdir(root_dir):
        if re.match(pattern, image_name):
            pass
        else:
            os.remove(image_name)
            print('remove {}'.format(image_name))


if __name__ == '__main__':
    # getimg(1000)
    # gettoken()
    # ocr()
    check()