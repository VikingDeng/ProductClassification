import argparse
import os

import src.dataset
import src.models
import src.runners

from src.core import RUNNERS
from src.utils import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--auto-resume', action='store_true')
    
    args = parser.parse_args()
    cfg = load_config(args.config)

    if 'workflows' in cfg:
        if args.mode not in cfg['workflows']:
            raise ValueError(f"Mode '{args.mode}' not defined in workflows")
        runner_cfg = cfg['workflows'][args.mode]
    else:
        runner_cfg = cfg['runner']

    resume_path = args.resume
    if args.mode == 'train' and args.auto_resume and resume_path is None:
        work_dir = runner_cfg.get('work_dir', './work_dirs')
        latest_path = os.path.join(work_dir, 'latest.pth')
        if os.path.exists(latest_path):
            print(f"[Auto-Resume] Found: {latest_path}")
            resume_path = latest_path

    print(f"[{args.mode.upper()}] Runner: {runner_cfg['type']}")

    runner = RUNNERS.build(runner_cfg)
    
    if args.mode == 'train':
        runner.run(cfg, resume_path=resume_path)
    else:
        runner.run(cfg)

if __name__ == '__main__':
    main()