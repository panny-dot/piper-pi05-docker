"""Offline PI0.5 checks and full-parameter training, without robot interfaces."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def environment(cpu=False):
    import torch
    from lerobot.policies.pi05 import PI05Config, PI05Policy
    from lerobot.scripts import lerobot_train
    import av
    result = {'python':sys.version.split()[0], 'torch':torch.__version__,
              'cuda_runtime':torch.version.cuda, 'method':'full-parameter; PEFT disabled',
              'packages':{k:metadata.version(k) for k in ['lerobot','transformers','accelerate','datasets','av','torchcodec']}}
    if not cpu:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; check the host NVIDIA driver and container runtime')
        tensor = torch.ones((16,16),device='cuda',dtype=torch.bfloat16)
        assert float((tensor @ tensor).float().mean()) == 16
        torch.cuda.synchronize()
        result['gpu'] = torch.cuda.get_device_name(0)
        result['gpu_total_bytes'] = torch.cuda.get_device_properties(0).total_memory
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def inspect_dataset(root):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = json.loads((root/'meta/info.json').read_text())
    assert info['fps']==30 and info['features']['action']['shape']==[14]
    assert info['features']['observation.state']['shape']==[14]
    dataset = LeRobotDataset('local/piper_train',root=root,video_backend='pyav',return_uint8=True)
    indices = sorted({0,len(dataset)//2,len(dataset)-1})
    for index in indices:
        row = dataset[index]
        for key in ['action','observation.state']:
            assert tuple(row[key].shape)==(14,) and bool(row[key].isfinite().all())
        for side in ['left','front','right']:
            assert tuple(row['observation.images.'+side].shape)==(3,480,640)
    result = {'episodes':info['total_episodes'],'frames':len(dataset),'sample_indices_checked':indices,
              'units':'joints rad; grippers m','action':'master targets','state':'puppet feedback'}
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['check','inspect-data','smoke','train'])
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--dataset',type=Path,default=Path('/data'))
    parser.add_argument('--model',type=Path,default=Path('/model'))
    parser.add_argument('--tokenizer',type=Path,default=Path('/tokenizer'))
    parser.add_argument('--output',type=Path,default=Path('/outputs'))
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--save-freq',type=int,default=5000)
    args = parser.parse_args()
    Path(os.environ['HOME']).mkdir(parents=True,exist_ok=True)
    environment(args.cpu)
    if args.mode=='check': return
    inspect_dataset(args.dataset)
    if args.mode=='inspect-data': return
    if args.cpu: parser.error('Full-parameter training requires CUDA in this configuration')
    for file in [args.model/'model.safetensors',args.model/'config.json',args.tokenizer/'tokenizer.json']:
        if not file.is_file(): raise FileNotFoundError(file)
    steps = args.steps if args.steps is not None else (2 if args.mode=='smoke' else 30000)
    batch = args.batch_size if args.batch_size is not None else 1
    if steps<=0 or batch<=0: parser.error('steps and batch size must be positive')
    if args.workers<0 or args.save_freq<=0: parser.error('workers must be nonnegative and save frequency must be positive')
    if args.mode=='smoke' and steps>5: parser.error('smoke is limited to 5 updates; use train for a long run')
    cfg = json.loads((Path(__file__).parent/'train_template.json').read_text())
    stamp = time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    job = 'pi05_full_'+args.mode+'_'+stamp
    args.output.mkdir(parents=True,exist_ok=True)
    cfg['dataset'].update(root=str(args.dataset),repo_id='local/piper_train',episodes=None,eval_split=0.0,video_backend='pyav')
    cfg['policy'].update(pretrained_path=str(args.model),pretrained_revision=None,device='cuda',dtype='bfloat16',
                         use_peft=False,freeze_vision_encoder=False,train_expert_only=False,
                         gradient_checkpointing=True,compile_model=False,push_to_hub=False)
    cfg.update(output_dir=str(args.output/job),job_name=job,resume=False,steps=steps,batch_size=batch,
               num_workers=args.workers,eval_steps=0,env_eval_freq=0,
               log_freq=1 if args.mode=='smoke' else 20,save_checkpoint=args.mode=='train',
               save_freq=args.save_freq,peft=None,checkpoint_path=None)
    config_path = args.output/(job+'_config.json')
    config_path.write_text(json.dumps(cfg,indent=2)+'\n')
    env = os.environ.copy()
    env.update(PIPER_RUN_ROOT=str(args.output),PIPER_AUDIT_DIR=str(args.output/(job+'_audit')),
               PIPER_TOKENIZER=str(args.tokenizer),PYTHONUNBUFFERED='1')
    print(json.dumps({'config':str(config_path),'output':cfg['output_dir'],'steps':steps,'batch_size':batch,
                      'use_peft':False,'freeze_vision_encoder':False,'train_expert_only':False,
                      'save_checkpoint':cfg['save_checkpoint']},indent=2),flush=True)
    result = subprocess.run([sys.executable,str(Path(__file__).parent/'train_entry.py'),
                             '--config_path='+str(config_path)],env=env)
    status={'exit_code':result.returncode,'mode':args.mode,'config':str(config_path),'output':cfg['output_dir']}
    (args.output/(job+'_status.json')).write_text(json.dumps(status,indent=2)+'\n')
    raise SystemExit(result.returncode)


if __name__=='__main__': main()
