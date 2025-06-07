import torchvision
import torch
from typing import Tuple, List

def get_dataset(name: str, root: str = "./data", batch: int = 8
               ) -> Tuple[torch.utils.data.Dataset,
                          torch.utils.data.DataLoader,
                          List[str]]:
    """
    name  : 'MNIST' | 'CIFAR10' | 'CIFAR100'
    return: (dataset, dataloader, class_names)
    """
    name_upper = name.upper()
    if name_upper == "MNIST":
        ds = torchvision.datasets.MNIST(root, train=False, download=True)
        classes = [str(i) for i in range(10)]
    elif name_upper == "CIFAR10":
        ds = torchvision.datasets.CIFAR10(root, train=False, download=True)
        classes = ds.classes
    elif name_upper == "CIFAR100":
        ds = torchvision.datasets.CIFAR100(root, train=False, download=True)
        classes = ds.classes
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    def collate_fn(batch):
        imgs, labels = zip(*batch)
        # print(labels)
        return list(imgs), list(labels)

    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=True, collate_fn=collate_fn
    )

    # classes: List[str] = ds.classes if hasattr(ds, "classes") else [str(i) for i in range(10)]
    # print(classes)
    return ds, dl, classes
