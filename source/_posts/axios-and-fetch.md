---
title: axios and fetch
date: 2023-11-24 12:26:37
tags: post
categories: Knowledge
---

# 0. 概述

axios是对XMLHttpRequest的封装，而Fetch是一种新的获取资源的接口方式，并不是对XMLHttpRequest的封装。

它们最大的不同点在于Fetch是浏览器原生支持，而Axios需要引入Axios库。

# 1. 基本用法

## 1.1 Get

![image-20231124123409179](axios-and-fetch/image-20231124123409179.png)

- axios可以根据headers里content-type自动转换。
- fetch需要手动对响应内容进行转换。

## 1.2 Post

json

![image-20231124123556652](axios-and-fetch/image-20231124123556652.png)

formdata

![image-20231124123635253](axios-and-fetch/image-20231124123635253.png)

## 1.3 数据流

![image-20231124123656813](axios-and-fetch/image-20231124123656813.png)

## 1.4 中止请求

![image-20231124123727305](axios-and-fetch/image-20231124123727305.png)

## 1.5 请求超时

![image-20231124123748576](axios-and-fetch/image-20231124123748576.png)

## 1.6 进度监控

![image-20231124123844465](axios-and-fetch/image-20231124123844465.png)

![image-20231124123854940](axios-and-fetch/image-20231124123854940.png)

# 2. 封装和配置

## 2.1 baseURL

![image-20231124124000816](axios-and-fetch/image-20231124124000816.png)

## 2.2 拦截器

![image-20231124124026371](axios-and-fetch/image-20231124124026371.png)

fetch需要自己封装

# 3. 兼容性与体积

![image-20231124124130175](axios-and-fetch/image-20231124124130175.png)

# 4. 总结

![image-20231124124154204](axios-and-fetch/image-20231124124154204.png)

