#!/usr/bin/python3
from os import path

import logging
log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

import subprocess as sp

try:
    from subprocess import DEVNULL 
except ImportError:
    DEVNULL = open(os.devnull, 'w')

def run_p(command):
    command_list = command.split()
    try:
        p = sp.Popen(command_list, stderr = DEVNULL, stdout = DEVNULL, stdin = DEVNULL)
        out, err = p.communicate()
        if err:
            print(err)
    finally:
        if p:
            p.kill()

def format_json(json_path):
    with open(json_path) as data:
        d = json.load(data)
        json_f = open(json_path, 'w')
        json.dump(d, json_f, indent = 4, sort_keys = True)

import urllib, urllib2, json
def getReport():
    param = {'': }
    url = ''
    data = urllib.urlencode(param)
    result = urllib2.urlopen(url,data)
    jdata =  json.loads(result.read())
    return jdata

def md5sum(filename):
    fh = open(filename, 'rb')
    m = hashlib.md5()
    while True:
        data = fh.read(8192)
        if not data:
            break
        m.update(data)
    return m.hexdigest()
    
#general program start with args

#!/usr/bin/python3
import argparse
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-', '--', default = '', type = ,
        required = , nargs='{#,*}', action = 'store_{true/false}', help = '')

    args = parser.parse_args()
    return args 

def main():
    args = get_args()

    start = time.time()


    ed = time.time() - start
    print("\nTraining/testing the model took: %s hours, %s min, %s sec\n" %
          (ed // (60 * 60), (ed % (60 * 60)) // 60, ed % 60))


if __name__ == '__main__':
    main()
