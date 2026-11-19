"""Launch LeRobot training with CPU and CUDA collectives for FSDP2 offload."""

import os
import runpy

import torch
import torch.distributed as dist


def main() -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    try:
        runpy.run_module(
            "lerobot.scripts.lerobot_train", run_name="__main__", alter_sys=True
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
