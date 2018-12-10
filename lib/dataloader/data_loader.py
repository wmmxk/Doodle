from torch.utils.data import DataLoader


class CreateLoader:
    def __init__(self):
        pass

    def __call__(self, dataset, shuffle=False,num_workers_=2,batch_size_=3):
        return DataLoader(
                   dataset=dataset,
                   shuffle=shuffle,
                   num_workers=num_workers,
                   batch_size=batch_size)
