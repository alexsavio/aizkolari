#!/usr/bin/python

import subprocess

def file_len(fname):
 p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
 result, err = p.communicate()
 if p.returncode != 0:
   raise IOError(err)
 l = result.strip()
 l = int(l.split()[0])
 return l
