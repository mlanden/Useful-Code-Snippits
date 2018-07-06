import errno
import os
from os import path, listdir
from os.path import isfile, join, isdir
import subprocess as sp

try:
    from subprocess import DEVNULL 
except ImportError:
    DEVNULL = open(os.devnull, 'w')


#create as many subdirectories that don't already exists
def mkdir_p(new_path):
    try:
        os.makedirs(new_path)
    except OSError as exc:  
        if exc.errno == errno.EEXIST and path.isdir(new_path):
            pass
        else:
            raise

#for file in get_files_of_type(root_dir, extention): ...
def get_files_of_type(root_dir, extention):
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(extention):
                 yield os.path.join(root, f)

def getDirsIn(dirName, joinDir=True):
    try:
        onlyDirs = [d for d in listdir(dirName) if isdir(join(dirName, d))]
        if joinDir:
            return [join(dirName, d) for d in onlyDirs]
        else:
            return onlyDirs
    except BaseException:
        raise

def write_lines(lines, file_name):
    with open(file_name, 'w') as out_file:
        text = '\n'.join(lines)
        out_file.write(text)

def format_json(json_path):
    with open(json_path) as data:
        d = json.load(data)
        json_f = open(json_path, 'w')
        json.dump(d, json_f, indent = 4, sort_keys = True)

def copy(src, dst):
    if not path.exists(dst):
        mkdir_p(dst)

    cp_cmd = "cp {} {}".format(src, dst)
    run_p(cp_cmd)

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

def get_imediate_dir(file_path):
    return path.basename(path.normpath(file_path))
