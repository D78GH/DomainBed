# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch
import os
from pathlib import Path

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


def slurm_files(commands):
    """
    Write one sbatch script per command.
    """

    output_dir = Path("slurm_jobs")
    output_dir.mkdir(exist_ok=True)

    header = """#!/bin/bash
#SBATCH --job-name=domainbed
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --exclude=damnii11,damnii08,damnii12,damnii10

module purge
module load cuda

source /home/s2457428/DomainBed/venv_domainbed/bin/activate

which python
module list
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

cd /home/s2457428/DomainBed/
export HF_TOKEN=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 
"""

    for i, command in enumerate(commands):
        script = output_dir / f"job_{i:04d}.sh"

        with open(script, "w") as f:
            f.write(header)
            f.write("\n")
            f.write(command)
            f.write("\n")

        os.chmod(script, 0o755)

    print(f"Wrote {len(commands)} sbatch scripts to {output_dir}")

def slurm(commands):
    partition = "Teaching"
    cpus_per_task = 1          # matches working script
    mem = "4G"                 # matches working script
    gres = "gpu:1"

    batch_size = 20
    submit_delay = 2
    batch_pause = 120

    for i, cmd in enumerate(commands):
        job_name = f"dbsweep_{i}"

        wrapped_cmd = "; ".join([
            "module purge",
            "module load cuda",
            "source /home/s2457428/DomainBed/venv_domainbed/bin/activate",
            "which python",
            "module list",
            'echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"',
            "cd /home/s2457428/DomainBed",
            cmd,
        ])

        sbatch = [
            "sbatch",
            f"--job-name={job_name}",
            f"--partition={partition}",
            f"--gres={gres}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={mem}",
            "--wrap",
            f"bash -lc '{wrapped_cmd}'",
        ]
        print(cmd)
        print(wrapped_cmd)
        print()
        print("Submitting:", " ".join(sbatch))
        subprocess.run(sbatch, check=True)

        if (i + 1) % batch_size == 0:
            print(f"Submitted {i + 1} jobs. Pausing {batch_pause}s...")
            time.sleep(batch_pause)
        else:
            time.sleep(submit_delay)

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'slurm': slurm,
    'slurm_files': slurm_files,
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
