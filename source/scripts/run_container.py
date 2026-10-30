#!/usr/bin/env python3
"""Host launcher: data/model/tokenizer read-only, outputs/cache writable."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['check','inspect-data','smoke','train'])
    p.add_argument('--image',default='piper-pi05-full:20261030')
    p.add_argument('--dataset',type=Path)
    p.add_argument('--model',type=Path)
    p.add_argument('--tokenizer',type=Path)
    p.add_argument('--output',type=Path,default=Path.cwd()/'pi05-runs')
    p.add_argument('--cache',type=Path)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--steps',type=int)
    p.add_argument('--save-freq',type=int)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    prefix=['docker']
    builder=os.environ.get('PIPER_DOCKER_EXEC_CONTAINER')
    if builder:prefix+=['exec',builder,'docker','-H','unix:///run/pi05/docker.sock']
    command=prefix+['run','--rm','--network','none','--shm-size','8g',
                    '--user',f'{os.getuid()}:{os.getgid()}']
    if not args.cpu:
        command+=['--device','nvidia.com/gpu=all'] if builder else ['--gpus','all']
    for key in ['dataset','model','tokenizer']:
        path=getattr(args,key)
        required=key=='dataset' and args.mode!='check' or key in ['model','tokenizer'] and args.mode in ['smoke','train']
        if required and path is None:p.error('--'+key+' is required for '+args.mode)
        if path is not None:
            path=path.expanduser().resolve()
            if not path.is_dir():p.error('Missing directory: '+str(path))
            command+=['--mount',f'type=bind,src={path},dst=/{"data" if key=="dataset" else key},readonly']
    output=args.output.expanduser().resolve()
    cache=args.cache.expanduser().resolve() if args.cache else output/'cache'
    if not args.dry_run:
        output.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    for path,target in [(output,'/outputs'),(cache,'/cache')]:
        command+=['--mount',f'type=bind,src={path},dst={target}']
    command+=[args.image,args.mode,'--workers',str(args.workers)]
    if args.cpu:command+=['--cpu']
    for name in ['batch_size','steps','save_freq']:
        if getattr(args,name) is not None:command+=['--'+name.replace('_','-'),str(getattr(args,name))]
    if args.dry_run:
        import shlex
        print(shlex.join(command));return
    raise SystemExit(subprocess.run(command).returncode)


if __name__=='__main__':main()
