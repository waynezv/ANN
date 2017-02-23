#!/bin/bash

# python spider.py -u "http://www.22mm.cc/" \
    # -o "./image1" -l 10000 \
    # -ae "jpg" \
    # -ad "meimei22.com"

# python spider.py -u "http://www.ivsky.com/tupian/dongwutupian/" \
    # -n 20 -o "./animals" -l 10000 -c "utf-8" -r 2 -t 0.001 \
    # -ae "jpg"

python spider.py -u "http://www.ivsky.com/tupian/zhiwuhuahui/" \
    -n 20 -o "./plants" -l 10000 -c "utf-8" -r 2 -t 0.001 \
    -ae "jpg"

python spider.py -u "http://www.ivsky.com/tupian/meishishijie/" \
    -n 20 -o "./food" -l 10000 -c "utf-8" -r 2 -t 0.001 \
    -ae "jpg"

python spider.py -u "http://www.ivsky.com/tupian/wupin/" \
    -n 20 -o "./objects" -l 10000 -c "utf-8" -r 2 -t 0.001 \
    -ae "jpg"

python spider.py -u "http://www.ivsky.com/tupian/jianzhuhuanjing/" \
    -n 20 -o "./building" -l 10000 -c "utf-8" -r 2 -t 0.001 \
    -ae "jpg"

python spider.py -u "http://www.tooopen.com/img/91.aspx" \
    -n 20 -o "./static" -l 10000 -c "utf-8" -r 2 -t 0.001 \
    -ae "jpg"

python spider.py -u "http://www.nipic.com/design/shengwu/index.html" \
    -n 20 -o "./life" -l 10000 -c "utf-8" -r 2 -t 0.001 \
    -ae "jpg"
