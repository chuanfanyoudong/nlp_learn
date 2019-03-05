#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: test.py
@time: 2019/3/5 14:08
"""
from _sha256 import sha256


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename
import requests
url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt"
response = requests.head(url, allow_redirects=True)
if response.status_code != 200:
    raise IOError("HEAD request failed for url {} with status code {}"
                  .format(url, response.status_code))
etag = response.headers.get("ETag")

filename = url_to_filename(url, etag)