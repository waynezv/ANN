#!usr/bin/python
# encoding=utf-8

import pycurl
import StringIO
from lxml import etree
import threading
import thread
import traceback
import time
import urlparse
import md5
from copy import deepcopy
import sys, os
import pdb
import argparse

def request(url, buffer):
    c = pycurl.Curl()
    c.setopt(pycurl.URL, url.encode(config.charset, 'ignore'))
    c.setopt(c.CONNECTTIMEOUT, 60)
    c.setopt(c.TIMEOUT, 600)
    c.setopt(c.WRITEFUNCTION, buffer.write)
    c.perform()


class Dispatcher(threading.Thread):
    def __init__(self, task):
        threading.Thread.__init__(self)
        self.task = task

    def run(self):
        while True:
            try:
                if self.task.dispatch_lock and len(threading.enumerate()) < config.max_thread_num and\
                len(self.task.to_crawl) > 0:
                    config.dispatch_sleep_time = float(config.dispatch_sleep_time/config.speed_ratio)
                    url = self.task.to_crawl.pop()
                    new_thread = threading.Thread(target=self.start_crawl, args=(url,))
                    new_thread.start()
                else:
                    config.dispatch_sleep_time = config.init_sleep_time
            except:
                traceback.print_exc()
            finally:
                time.sleep(config.dispatch_sleep_time)

    def start_crawl(self, url):
        buffer = StringIO.StringIO()
        request(url, buffer)
        html = buffer.getvalue()
        self.task.to_filt.extend(self.start_parse(html))

    def start_parse(self, html):
        result = set()
        parser = etree.HTMLParser()
        if html:
            try:
                tree = etree.parse(StringIO.StringIO(html.lower().decode(config.charset, 'ignore')), parser)
            except:
                tree = etree.parse(StringIO.StringIO(html.lower()), parser)

            for a in tree.xpath("//img"):
                result.add(a.get('src'))
            for a in tree.xpath("//a"):
                result.add(a.get('href'))
        return result


class Collector(threading.Thread):
    def __init__(self, task):
        threading.Thread.__init__(self)
        self.task = task
        self.mylock = threading.Lock()

    def run(self):
        while True:
            try:
                if self.task.current_picture_num >= config.max_picture_num and \
                   config.max_picture_num != -1:
                    self.task.to_crawl = []
                    self.task.to_filt = []
                elif self.task.collect_lock and \
                   len(threading.enumerate()) < config.max_thread_num and\
                   len(self.task.to_filt) > 0:
                    config.collect_sleep_time = float(config.collect_sleep_time/config.speed_ratio)
                    url = self.task.to_filt.pop()

                    new_thread = threading.Thread(target=self.start_filt, args=(url,))
                    new_thread.start()
                else:
                    config.collect_sleep_time = config.init_sleep_time
            except:
                traceback.print_exc()
            finally:
                time.sleep(config.collect_sleep_time)

    def store_picture(self, url, file_ext):
        self.mylock.acquire()
        self.task.current_picture_num += 1
        print "get total %d pictures\n" % self.task.current_picture_num
        file_name = str(self.task.current_picture_num) + '.' + file_ext
        path = os.path.join(config.store_path, file_name)
        self.mylock.release()
        f = open(path, "wb")
        request(url, f)
        f.close()

    def is_duplicate(self, url):
        m = md5.new()
        m.update(url.encode(config.charset, 'ignore'))
        key = m.hexdigest()
        if key in self.task.duplicate_set:
            return True
        else:
            self.task.duplicate_set.add(key)
            return False

    def is_allow_domain(self, netloc_normal, sub_domain):
        if netloc_normal.endswith(sub_domain) or config.crawl_external or \
           self.allow_domain(netloc_normal):
            return True
        else:
            return False

    def allow_domain(self, netloc_normal):
        for domain in config.white_domain:
            if netloc_normal.endswith(domain):
                return True
        return False

    def start_filt(self, url):
        if url:
            parsed_normal_url = urlparse.urlparse(url)
            parsed_origin_url = urlparse.urlparse(config.origin_url)
            netloc_origin = parsed_origin_url[1]
            if not parsed_normal_url[0]:
                url = urlparse.urlunparse((parsed_origin_url[0], netloc_origin, parsed_normal_url[2],\
                                          parsed_normal_url[3] ,parsed_normal_url[4] ,parsed_normal_url[5] ))
            parsed_normal_url = urlparse.urlparse(url)
            netloc_normal = parsed_normal_url[1]
            position = netloc_origin.find('.')
            domain = netloc_origin[:position]
            if domain in ["www"]:
                sub_domain = netloc_origin[position:]
            else:
                sub_domain = netloc_origin

            if self.is_allow_domain(netloc_normal, sub_domain):
                if not self.is_duplicate(url):
                    path = parsed_normal_url[2]
                    file_ext = path.split('.')[-1]
                    if file_ext in config.white_ext:
                        self.store_picture(url, file_ext)
                    else:
                        self.task.to_crawl.append(url)
                else:
                    #print "duplicate"
                    pass
            else:
                #print "external"
                pass


class Config(object):
    def __init__(self):
        self.charset = 'utf-8'
        self.init_sleep_time = 0.001
        self.speed_ratio = 1
        self.collect_sleep_time = self.init_sleep_time
        self.dispatch_sleep_time = self.init_sleep_time
        self.crawl_external = False
        self.origin_url = None
        self.max_thread_num = 10
        self.max_picture_num = -1
        self.store_path = os.path.join(os.path.realpath('.'), 'pic')
        self.white_ext = ['jpg', 'png', 'gif']
        self.white_domain = ['meimei22.com']

    def init(self, config):
        self.origin_url = config.origin_url
        self.max_thread_num = config.max_thread_num
        self.max_picture_num = config.max_picture_num
        self.store_path = config.picture_path
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)
        self.charset = config.charset
        self.crawl_external = config.crawl_external
        self.speed_ratio = config.speed_ratio
        self.init_sleep_time = config.init_sleep_time
        self.white_ext = config.white_ext
        self.white_domain = config.white_domain


class Task(object):
    def __init__(self):
        self.to_crawl = []
        self.to_filt = []
        self.duplicate_set = set()
        self.current_picture_num = 0
        self.dispatch_lock = True
        self.collect_lock = True
        self.tick = 0

    def is_finished(self):
        if len(self.to_crawl) == 0 and len(self.to_filt) == 0:
            self.tick += 1
        else:
            self.tick = 0
        if self.tick > 60:
            print "task is finished totally crawl %s urls\n" % self.current_picture_num
            return True
        return False

    def auto_regulate(self):
        print "current thread num %s,to_crawl %s, to_filt %s\n" % (len(threading.enumerate()), \
                                                                   len(self.to_crawl), \
                                                                   len(self.to_filt))
        if len(self.to_crawl) > len(self.to_filt):
            self.dispatch_lock = True
            self.collect_lock = False
        else:
            self.dispatch_lock = False
            self.collect_lock = True


def parse_argument(parser):
    parser.add_argument('-u', action='store', dest='origin_url', \
                        required=True, help='the url to crawl')
    parser.add_argument('-n', action='store', dest='max_thread_num', \
                        default=10, type=int, help='max thread number')
    parser.add_argument('-o', action='store', dest='picture_path', \
                        default=os.path.join(os.path.realpath('.'), 'pic'), \
                        help='pictures store path')
    parser.add_argument('-l', action='store', dest='max_picture_num', \
                        default=-1, type=int, help='the max picture number to crawl')
    parser.add_argument('-c', action='store', dest='charset', \
                        default='utf-8', help='charset')
    parser.add_argument('-e', action='store_true', dest='crawl_external', default=False, \
                        help='crawl external link or not')
    parser.add_argument('-r', action='store', dest='speed_ratio', default=1, \
                        type=float, help='the speed up ratio')
    parser.add_argument('-t', action='store', dest='init_sleep_time', default=0.001, \
                        type=float, help='thread sleep time')
    parser.add_argument('-ae', action='append', dest='white_ext', default=['jpg', 'png', 'gif'], \
                        type=str, help='picture type you want to crawl, example:-ae "jpg" -ap "png"')
    parser.add_argument('-ad', action='append', dest='white_domain', default=['meimei22.com'], \
                        type=str, help='domain you want to crawl, example:-ad "test.com" -ad "test1.com"')


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parse_argument(parser)
    results = parser.parse_args()
    task = Task()
    config = Config()
    if results.speed_ratio == 0 or results.init_sleep_time == 0:
        print "speed up ratio or thread sleep time can not be zero!"
        exit(0)
    config.init(results)
    task.to_crawl.append(results.origin_url)
    dispatcher = Dispatcher(task)
    dispatcher.setDaemon(True)
    dispatcher.start()
    collector = Collector(task)
    collector.setDaemon(True)
    collector.start()

    while True:
        task.auto_regulate()
        if task.is_finished():
            print "used time %s" % (time.time()-start_time)
            break
        time.sleep(1)
