#coding:utf-8
"""
Created on 2015-9-17

@author: huangxie
"""
import time,math,os,re,urllib,urllib2,cookielib 
from bs4 import BeautifulSoup
import time  
import re
import uuid
import json
from threading import Thread
from Queue import Queue 
import MySQLdb as mdb
import sys
import threading
import utils
import imitate_browser
from MySQLdb.constants.REFRESH import STATUS
reload(sys)
sys.setdefaultencoding('utf-8')

DB_HOST = '127.0.0.1'
DB_USER = 'root'
DB_PASS = 'root'
proxy = {u'http':u'222.39.64.13:8118'}
TOP_URL="http://image.baidu.com/i?tn=resultjsonavatarnew&ie=utf-8&word={word}&pn={pn}&rn={rn}"
KEYWORD_URL="https://www.baidu.com/s?ie=utf-8&f=8&tn=baidu&wd={wd}"

"""
i_headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
              'Accept':'json;q=0.9,*/*;q=0.8',
              'Accept-Charset':'utf-8;q=0.7,*;q=0.3',
              'Accept-Encoding':'gzip',
              'Connection':'close',
              'Referer':None #注意如果依然不能抓取的话，这里可以设置抓取网站的host
            }
"""
i_headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1650.48'}

def GetDateString():
    x = time.localtime(time.time())
    foldername = str(x.__getattribute__("tm_year"))+"-"+str(x.__getattribute__("tm_mon"))+"-"+str(x.__getattribute__("tm_mday"))
    return foldername 

class BaiduImage(threading.Thread):     

    def __init__(self):
        Thread.__init__(self)
        self.browser=imitate_browser.BrowserBase()
        self.chance=0
        self.chance1=0
        self.request_queue=Queue()
        self.wait_ana_queue=Queue()
        #self.key_word_queue.put((("动态图", 0, 24)))
        self.count=0
        self.mutex = threading.RLock() #可重入锁，使单线程可以再次获得已经获得的锁
        self.commit_count=0
        self.ID=500
        self.next_proxy_set = set()
        self.dbconn = mdb.connect(DB_HOST, DB_USER, DB_PASS, 'sosogif', charset='utf8')
        self.dbconn.autocommit(False)
        self.dbcurr = self.dbconn.cursor()
        self.dbcurr.execute('SET NAMES utf8')
        
    """
    def run(self):
        while True:
            self.get_pic()
    """
    
    def work(self,item):
        print "start thread",item
        while True: #MAX_REQUEST条以上则等待
            self.get_pic()
            self.prepare_request()
    
    def format_keyword_url(self,keyword):
  
        return KEYWORD_URL.format(wd=keyword).encode('utf-8') 
           
    def generateSeed(self,url):
        
        html = self.browser.openurl(url).read()
        if html:
            try:
                soup = BeautifulSoup(html)
                trs = soup.find('div', id='rs').find('table').find_all('tr') #获得所有行
                for tr in trs:
                    ths=tr.find_all('th')
                    for th in ths:
                        a=th.find_all('a')[0]
                        keyword=a.text.strip()
                        if "动态图" in keyword or "gif" in keyword:
                            print "keyword",keyword
                            self.dbcurr.execute('select id from info where word=%s',(keyword))
                            y = self.dbcurr.fetchone()
                            if not y:
                                self.dbcurr.execute('INSERT INTO info(word,status,page_num,left_num,how_many) VALUES(%s,0,0,0,0)',(keyword))
                    self.dbconn.commit()
            except:
                pass
                
               
    def prepare_request(self):
        self.lock()
        self.dbcurr.execute('select * from info where status=0')
        result = self.dbcurr.fetchone()
        if result:
            id,word,status,page_num,left_num,how_many=result
            self.request_queue.put((id,word,page_num)) 
            if page_num==0 and left_num==0 and how_many==0:
                url=self.format_keyword_url(word)
                self.generateSeed(url)
                html=""
                try:
                    url=self.format_top_url(word, page_num, 24)
                    html = self.browser.openurl(url).read()
                except Exception as err:
                    print "err",err
                    #pass
                if html!="":
                    how_many=self.how_many(html)
                    print "how_many",how_many
                    if how_many==None:
                        how_many=0
                    t=math.ceil(how_many/24*100) #只要前1/100即可
                    num = int(t)
                    for i  in xrange(0,num-1):
                        self.dbcurr.execute('INSERT INTO info(word,status,page_num,left_num,how_many) VALUES(%s,%s,%s,%s,%s)',(word,0,i*24,num-i,how_many))
                    self.dbcurr.execute('update info SET status=1 WHERE id=%s',(id)) #置为已经访问
                    self.dbconn.commit()
        self.unlock()
                
            
    def start_work(self,req_max):
        for item in xrange(req_max):
            t = threading.Thread(target=self.work, args=(item,))
            t.setDaemon(True)
            t.start()
            
    def lock(self): #加锁
        self.mutex.acquire()

    def unlock(self): #解锁
        self.mutex.release()

    def get_para(self,url,key):
        values = url.split('?')[-1]
        for key_value in values.split('&'):
            value=key_value.split('=')
            if value[0]==key:
                return value[1]
        return None  
    
    def makeDateFolder( self,par,child):
        #self.lock()
        if os.path.isdir( par ):
            path=par + '//' + GetDateString()
            newFolderName = path+'//'+child
            if not os.path.isdir(path):
                os.mkdir(path)
            if not os.path.isdir( newFolderName ):
                os.mkdir( newFolderName )
            return newFolderName
        else:
            return par 
        #self.unlock()
        
    def parse_json(self,data):
        
        ipdata = json.loads(data)
        try:
            if ipdata['imgs']:  
                for n in ipdata['imgs']: #data子项 
                    if n['objURL']:  
                        try:
                            proxy_support = urllib2.ProxyHandler(proxy)
                            opener = urllib2.build_opener(proxy_support)
                            urllib2.install_opener(opener)
                            #print "proxy",proxy
                            self.lock()
                            self.dbcurr.execute('select ID from pic_info where objURL=%s', (n['objURL']))
                            y = self.dbcurr.fetchone()
                            #print "y=",y
                            if y:
                                print "database exist"
                                self.unlock() #continue 前解锁
                                continue
                            else:
                                real_extension=utils.get_extension(n['objURL'])
                                req = urllib2.Request(n['objURL'],headers=i_headers)
                                resp = urllib2.urlopen(req,None,5)
                                dataimg=resp.read()
                                name=str(uuid.uuid1())
                                filename=""
                                if len(real_extension)>4:
                                    real_extension=".gif"
                                real_extension=real_extension.lower()
                                if real_extension==".gif":
                                    filename  =self.makeDateFolder("E://sosogif", "d"+str(self.count % 60))+"//"+name+"-www.sosogif.com-搜搜gif贡献"+real_extension
                                    self.count+=1
                                else:
                                    filename  =self.makeDateFolder("E://sosogif", "o"+str(self.count % 20))+"//"+name+"-www.sosogif.com-搜搜gif贡献"+real_extension
                                    self.count+=1
                                """
                                name=str(uuid.uuid1())
                                filename=""
                                if len(real_extension)>4:
                                    real_extension=".gif"
                                filename  =self.makeDateFolder("E://sosogif", "d"+str(self.count % 60))+"//"+name+"-www.sosogif.com-搜搜gif贡献"+real_extension
                                self.count+=1 
                                """
                                try: 
                                    if not os.path.exists(filename): 
                                        file_object = open(filename,'w+b')  
                                        file_object.write(dataimg)  
                                        file_object.close()
                                        self.anaylis_info(n,filename,real_extension) #入库操作
                                    else:
                                        print "file exist" 
                                except IOError,e1:  
                                    print "e1=",e1
                                    pass
                            self.unlock()
                        except IOError,e2:  
                            #print "e2=",e2 
                            pass  
                            self.chance1+=1
        except Exception as parse_error:
            print "parse_error",parse_error
            pass
    
    def title_dealwith(self,title):
        
        #print "title",title
        a=title.find("<strong>")
        temp1=title[0:a]
        b=title.find("</strong>")
        temp2=title[a+8:b]
        temp3=title[b+9:len(title)]
        return (temp1+temp2+temp3).strip()
        
    def anaylis_info(self,n,filename,real_extension):
        print "success."
        
        #if self.wait_ana_queue.qsize()!=0:
            #n,filename,real_extension=self.wait.ana_queue.get()
        #self.lock()
        objURL=n['objURL'] #图片地址
        fromURLHost=n['fromURLHost'] #来源网站
        width=n['width']  #宽度
        height=n['height'] #高度
        di=n['di'] #用来唯一标识
        type=n['type'] #格式
        fromPageTitle=n['fromPageTitle'] #来自网站
        keyword=self.title_dealwith(fromPageTitle)
        cs=n['cs'] #未知
        os=n['os'] #未知
        temp = time.time()
        x = time.localtime(float(temp))
        acTime = time.strftime("%Y-%m-%d %H:%M:%S",x) #爬取时间
        self.dbcurr.execute('select ID from pic_info where cs=%s', (cs))
        y = self.dbcurr.fetchone()
        if not y:
            print 'add pic',filename
            self.commit_count+=1
            self.dbcurr.execute('INSERT INTO pic_info(objURL,fromURLHost,width,height,di,type,keyword,cs,os,acTime,filename,real_extension) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',(objURL,fromURLHost,width,height,di,type,keyword,cs,os,acTime,filename,real_extension))
            if self.commit_count==10:
                self.dbconn.commit()
                self.commit_count=0
        #self.unlock()
           

    def format_top_url(self,word,pn,rn):

        url = TOP_URL.format(word=word, pn=pn,rn=rn).encode('utf-8') 
        return url

    def how_many(self,data):
        try:
            ipdata = json.loads(data)
            if ipdata['displayNum']>0:
                how_many=ipdata['displayNum']
                return int(how_many)
            else:
                return 0
        except Exception as e:
            pass
        
    def get_pic(self):
        """
        word="gif"
        pn=0
        rn=24
        if self.key_word_queue.qsize()!=0:
            word,pn,rn=self.key_word_queue.get()
        url=self.format_top_url(word,pn,rn)
        global proxy
        if url:
            try:
                html=""
                try:
                    req = urllib2.Request(url,headers=i_headers)
                    response = urllib2.urlopen(req, None,5)
                    #print "url",url
                    html = self.browser.openurl(url).read()
                except Exception as err:
                    print "err",err
                    #pass
                if html:
                    how_many=self.how_many(html)
                    #how_many=10000
                    print "how_many",how_many
                    word=self.get_para(url,"word")
                    rn=int(self.get_para(url,"rn"))
                    t=math.ceil(how_many/rn)
                    num = int(t)
                    for item  in xrange(0,num-1):
        """
        try:
            global proxy
            print "size of queue",self.request_queue.qsize()
            if self.request_queue.qsize()!=0:
                id,word,page_num = self.request_queue.get()            
                u=self.format_top_url(word,page_num,24)
                self.lock()
                self.dbcurr.execute('update info SET status=1 WHERE id=%s',(id))
                self.dbconn.commit()
                if self.chance >0 or self.chance1>1: #任何一个出问题都给换代理
                    if self.ID % 100==0:
                        self.dbcurr.execute("select count(*) from proxy")
                        for r in self.dbcurr:
                            count=r[0]
                        if self.ID>count:
                            self.ID=50
                    self.dbcurr.execute("select * from proxy where ID=%s",(self.ID))
                    results = self.dbcurr.fetchall()
                    for r in results:
                        protocol=r[1]
                        ip=r[2]
                        port=r[3]
                        pro=(protocol,ip+":"+port)
                        if pro not in self.next_proxy_set:
                            self.next_proxy_set.add(pro)
                    self.chance=0
                    self.chance1=0
                    self.ID+=1
                self.unlock() 
                proxy_support = urllib2.ProxyHandler(proxy)
                opener = urllib2.build_opener(proxy_support)
                urllib2.install_opener(opener)
                html=""
                try:
                    req = urllib2.Request(u,headers=i_headers)
                    #print "u=",u
                    response = urllib2.urlopen(req, None,5)
                    html = response.read()
                    if html:
                        #print "html",type(html)
                        self.parse_json(html)
                except Exception as ex1:
                    #print "error=",ex1
                    pass
                    self.chance+=1
                    if self.chance>0 or self.chance1>1:
                        if len(self.next_proxy_set)>0:
                            protocol,socket=self.next_proxy_set.pop()
                            proxy= {protocol:socket}
                            print "change proxy finished<<",proxy,self.ID
        except Exception as e:
            print "error1",e
            pass
            
if __name__ == '__main__':

    app = BaiduImage() 
    app.start_work(80)
    #app.generateSeed()
    while 1:
        pass
