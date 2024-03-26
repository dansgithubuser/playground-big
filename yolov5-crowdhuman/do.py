#! /usr/bin/env python3

#===== imports =====#
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

#===== args =====#
parser = argparse.ArgumentParser()
parser.add_argument('--setup', action='store_true')
parser.add_argument('--test-basic', action='store_true', help='detect heads in data/images')
parser.add_argument('--detect', metavar='path', help='detect heads in folder containing images or videos')
args = parser.parse_args()

#===== consts =====#
DIR = Path(__file__).resolve().parent

#===== setup =====#
os.chdir(DIR)

#===== helpers =====#
def blue(text):
    return '\x1b[34m' + text + '\x1b[0m'

def timestamp():
    return datetime.now().astimezone().isoformat(' ', 'seconds')

def invoke(
    *args,
    quiet=False,
    env_add={},
    env_add_secrets=set(),
    handle_sigint=True,
    popen=False,
    check=True,
    put_in=False,
    get_out=False,
    get_err=False,
    **kwargs,
):
    if len(args) == 1 and type(args[0]) == str and not kwargs.get('shell'):
        args = args[0].split()
    if not quiet:
        print(blue('-'*40))
        print(timestamp())
        print(f'{Path.cwd()}$', end=' ')
        if any([re.search(r'\s', i) for i in args]):
            print()
            for i in args: print(f'\t{i} \\')
        else:
            for i, v in enumerate(args):
                if i != len(args)-1:
                    end = ' '
                else:
                    end = ';\n'
                print(v, end=end)
        if env_add:
            print('env_add:', {k: (v if k not in env_add_secrets else '...') for k, v in env_add.items()})
        if kwargs: print(kwargs)
        if popen: print('popen')
        print()
    if env_add:
        env = os.environ.copy()
        env.update(env_add)
        kwargs['env'] = env
    if put_in and 'stdin' not in kwargs: kwargs['stdin'] = subprocess.PIPE
    if get_out: kwargs['stdout'] = subprocess.PIPE
    if get_err: kwargs['stderr'] = subprocess.PIPE
    p = subprocess.Popen(args, **kwargs)
    if handle_sigint:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if put_in:
        if type(put_in) == str:
            put_in = put_in.encode()
        p.stdin.write(put_in)
        if popen:
            p.stdin.flush()
    if popen:
        return p
    stdout, stderr = p.communicate()
    p.out = stdout
    p.err = stderr
    if handle_sigint:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    if check and p.returncode:
        e = Exception(f'invocation {repr(args)} returned code {p.returncode}.')
        e.p = p
        raise e
    if get_out:
        stdout = stdout.decode('utf-8')
        if get_out != 'exact': stdout = stdout.strip()
        if not get_err: return stdout
    if get_err:
        stderr = stderr.decode('utf-8')
        if get_err != 'exact': stderr = stderr.strip()
        if not get_out: return stderr
    if get_out and get_err: return stdout, stderr
    return p

def invoke_in_venv(invocation):
    return invoke(f'. venv/bin/activate && {invocation}', shell=True)

#===== main =====#
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

if args.setup:
    invoke('python3 -m venv venv')
    invoke_in_venv('pip3 install -r requirements.frozen.txt')

if args.test_basic:
    invoke_in_venv('python3 detect.py --weights crowdhuman_yolov5m.pt --source data/images --heads --view-img')

if args.detect:
    invoke_in_venv(f'python3 detect.py --weights crowdhuman_yolov5m.pt --source {args.detect} --heads --view-img')
