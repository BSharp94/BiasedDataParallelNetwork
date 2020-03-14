import numpy as np
import os
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.multiprocessing import Process
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from random import Random

WORLD_SIZE = 2
NUM_EPOCHS = 50
TRAINING_RECORD_INTERVAL = 25

class Partition(object):
    
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes, seed = 8675309):
        self.data = data
        self.partions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partion):
        return Partition(self.data, self.partions[partion])

def partition_dataset():
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(244),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    dataset = datasets.CIFAR10('./data', download = True, train = True, 
        transform = transform_train)

    size = dist.get_world_size()
    bsz = 128 // float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=int(bsz),
                                         shuffle=True)
    return train_set, bsz

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def run(rank, size, model, criterion, optimizer):
    
    torch.manual_seed(8675309)
    training_set, bsz = partition_dataset()

    # Set up record holder and testing set for only the master node
    if rank == 0:
        training_accuracy = []
        testing_accuracy = []
        
        transform_test = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        imagenet_data_test = datasets.CIFAR10('./data', download = True, train = False, transform = transform_test)
        test_loader = torch.utils.data.DataLoader(imagenet_data_test, batch_size=64,
                                          shuffle=False, num_workers=0)


    for epoch_idx in range(NUM_EPOCHS):

        for batch_idx, (data, target) in enumerate(training_set):
            optimizer.zero_grad()
            outputs = model(data)
            
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == target).sum().item()
            loss = criterion(outputs, target)

            print('Rank %d\tEpoch: %d\tIterval: %d\tAccuracy : %d%%' % (
                rank,
                epoch_idx,
                batch_idx,
                100 * correct / 64))

            loss.backward()
        
            if rank == 0:
                training_accuracy.append(100 * correct / 64)

            average_gradients(model)
            optimizer.step()
        
        # After each epoch record testing results on master node
        if rank == 0:
            testing_correct = 0
            for idx, (inputs, labels) in enumerate(test_loader):

                inputs, labels = inputs, labels
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                testing_correct += (predicted == labels).sum().item()
        
            print('Epoch: %d\tAccuracy: %d %%' % (epoch_idx, 100 * testing_correct / len(imagenet_data_test)))
            testing_accuracy.append(100 * testing_correct / len(imagenet_data_test))

        # wait till all processes have finished the epoch
        dist.barrier()

    if rank == 0:
        np.save('./training_accuracy.npy', training_accuracy)
        np.save('./testing_accuracy.npy', testing_accuracy)
        torch.save(model.state_dict(), './model_control.pt')

def init_process(rank, size, model, criterion, optimizer, fn, backend = "gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank = rank, world_size = size)
    fn(rank, size, model, criterion, optimizer)

if __name__ == "__main__":
    processes = []
    model = models.alexnet(num_classes = 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
    
    for rank in range(WORLD_SIZE):
        p = Process(target = init_process, args = (rank, WORLD_SIZE, model, criterion, optimizer, run))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    print("Execution Finished")