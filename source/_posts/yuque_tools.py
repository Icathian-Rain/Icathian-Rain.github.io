# python main.py [origin_md_path] [output_md_path] [image_dir] [image_url_prefix] [image_rename_mode]
# origin_md_path: 输入的markdown文件路径
# output_md_path: 输出的markdown文件路径
# image_dir: 图片存储的目录
# image_url_prefix: 图片链接前缀，空字符串或者路径或者CDN地址
# image_rename_mode: 图片重命名模式，raw: 原始uuid模式，asc: 递增重命名模式
# python .\yuque_tools.py ./lab1.md ./mit-6583-lab1.md 

import re
import requests
import os
import sys
import time

yuque_cdn_domain = 'cdn.nlark.com'
output_content = []
image_file_prefix = 'image-'

file_name = ""

def main():

    origin_md_path = sys.argv[1]
    output_md_path = sys.argv[2]
    global file_name
    file_name = output_md_path.split('/')[-1].split('.')[0]    

    image_dir = './' + file_name + '/'
    image_url_prefix = './' + file_name + '/'
    image_rename_mode = 'raw'
    
    mkdir(image_dir)
    cnt = handler(origin_md_path, output_md_path, image_dir, image_url_prefix, image_rename_mode)
    print('处理完成, 共{}张图片'.format(cnt))


def mkdir(image_dir):
    isExists = os.path.exists(image_dir)
    if isExists:
        print('图片存储目录已存在')
    else:
        os.makedirs(image_dir)
        print('图片存储目录创建成功')


def handler(origin_md_path, output_md_path, image_dir, image_url_prefix, image_rename_mode):
    idx = 0
    with open(origin_md_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            line = re.sub(r'png#(.*)+', 'png)', line)
            image_url = str(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',line))
            if yuque_cdn_domain in image_url:
                image_url = image_url.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace("'", '')
                if '.png' in image_url:
                    suffix = '.png'
                elif '.jpeg' in image_url:
                    suffix = '.jpeg'
                download_image(image_url, image_dir, image_rename_mode, idx, suffix)
                to_replace = '/'.join(image_url.split('/')[:-1])
                new_image_url = image_url.replace(to_replace, 'placeholder')
                if image_rename_mode == 'asc':
                    new_image_url = image_url_prefix + image_file_prefix + str(idx) + suffix
                else:
                    new_image_url = new_image_url.replace('placeholder/',image_url_prefix)
                idx += 1
                line = line.replace(image_url, new_image_url)
            output_content.append(line)
    prefix_content = f'''---
title: {file_name.replace('-', ' ')}
date: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
tags: 
categories: 
---
'''.format(output_md_path.split('/')[-1].split('.')[0], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with open(output_md_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(prefix_content)
        for _output_content in output_content:
            f.write(str(_output_content))
    return idx
            

def download_image(image_url, image_dir, image_name_mode, idx, suffix):
    r = requests.get(image_url, stream=True)
    image_name = image_url.split('/')[-1]
    if image_name_mode == 'asc':
        image_name = image_file_prefix + str(idx) + suffix
    if r.status_code == 200:
        open(image_dir+'/'+image_name, 'wb').write(r.content)
    del r

if __name__ == '__main__':
    main()