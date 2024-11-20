import torch.nn as nn
# Here we use DistributedDataParallel(DDP) rather than DataParallel(DP) for multiple GPUs training


def is_multi_gpu(net):
    return isinstance(net, (MultiGPU, nn.parallel.distributed.DistributedDataParallel))


class MultiGPU(nn.parallel.distributed.DistributedDataParallel):
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)
