import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    print(rank)
    torch.manual_seed(rank)
    device_id = rank % torch.cuda.device_count()
    model = MyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)
    for i in range(2):
        outputs = ddp_model(torch.randn(20, 10).to(device_id))
        labels = torch.randn(20, 5).to(device_id)
        optimizer.zero_grad()
        loss_fn(outputs, labels).backward()
        if rank == 0:
            print("rank", rank, "before step", model.net2.weight, )
        optimizer.step()
        if rank == 0:
            print("rank", rank, "after step", model.net2.weight, )
    dist.destroy_process_group()


if __name__ == "__main__":
    demo_basic()


'''
export CUDA_VISIBLE_DEVICES="0,1,2,3"
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 702_ddp_torchrun.py

'''