import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.data import random_split

from .basic_dataset import FedDataset, Subset
from .partition import  MNISTPartitioner, FMNISTPartitioner
from torch.utils.data.sampler import SubsetRandomSampler

class PartitionedFEMNIST(FedDataset):
    """:class:`FedDataset` with partitioning preprocess. For detailed partitioning, please
    check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.


    Args:
        root (str): Path to download raw dataset.
        path (str): Path to save partitioned subdataset.
        num_clients (int): Number of clients.
        download (bool): Whether to download the raw dataset.
        preprocess (bool): Whether to preprocess the dataset.
        partition (str, optional): Partition name. Only supports ``"noniid-#label"``, ``"noniid-labeldir"``, ``"unbalance"`` and ``"iid"`` partition schemes.
        dir_alpha (float, optional): Dirichlet distribution parameter for non-iid partition. Only works if ``partition="dirichlet"``. Default as ``None``.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.
        seed (int, optional): Random seed. Default as ``None``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self,
                 root,
                 path,
                 num_clients,
                 major_classes_num,
                 download=True,
                 preprocess=False,
                 partition="iid",
                 dir_alpha=None,
                 verbose=True,
                 seed=None,
                 transform=None,
                 target_transform=None) -> None:

        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.transform = transform
        self.targt_transform = target_transform
        self.major_classes_num = major_classes_num

        if preprocess:
            self.preprocess(partition=partition,
                            dir_alpha=dir_alpha,
                            major_classes_num = major_classes_num,
                            verbose=verbose,
                            seed=seed,
                            download=download,
                            transform=transform,
                            target_transform=target_transform)

    def preprocess(self,
                   partition="iid",
                   dir_alpha=None,
                   major_classes_num=10,
                   verbose=True,
                   seed=None,
                   download=True,
                   transform=None,
                   target_transform=None):
        """Perform FL partition on the dataset, and save each subset for each client into ``data{cid}.pkl`` file.

        For details of partition schemes, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
        """
        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "val"))
            os.mkdir(os.path.join(self.path, "test"))

        trainset = torchvision.datasets.FashionMNIST(root=self.root,
                                              train=True,
                                              download=download)
        testset = torchvision.datasets.FashionMNIST(root=self.root,
                                                    train=False,
                                                    download=download)

        print("Trainset size:", len(trainset))
        print("Testset size:", len(testset))


        # get 0.1 from the training set
        split = int(np.floor(0.9 * len(trainset)))
        train_indices = range(split)
        valid_indices = range(split, len(trainset))
        train_set = torch.utils.data.Subset(trainset, train_indices)
        valid_set = torch.utils.data.Subset(trainset, valid_indices)



        # Access the targets from the subsets
        train_targets = [trainset.targets[i] for i in train_indices]
        valid_targets = [trainset.targets[i] for i in valid_indices]
        test_targets = testset.targets

        # Print the sizes of the resulting datasets
        print("Train_set target size {} and type {}:".format(len(train_targets), type(train_targets)))
        print("Valid_set target size {} and type {}: ".format(len(valid_targets), type(valid_targets)))
        print("test_set target size {} and type {}: ".format(len(test_targets), type(test_targets)))

        partitioner = FMNISTPartitioner(train_targets,
                                       self.num_clients,
                                       partition=partition,
                                       dir_alpha=dir_alpha,
                                       major_classes_num = major_classes_num,
                                       verbose=verbose,
                                       seed=seed)

        partitioner_val = FMNISTPartitioner(valid_targets,
                                       self.num_clients,
                                       partition=partition,
                                       dir_alpha=dir_alpha,
                                       major_classes_num=major_classes_num,
                                       verbose=verbose,
                                       seed=seed)

        partitioner_test = FMNISTPartitioner(testset.targets,
                                       self.num_clients,
                                       partition=partition,
                                       dir_alpha=dir_alpha,
                                       major_classes_num=major_classes_num,
                                       verbose=verbose,
                                       seed=seed)

        train_data = trainset.data[train_indices]
        valid_data = trainset.data[valid_indices]
        test_data = testset.data

        # Print the sizes of the resulting datasets
        print("Train_set data size {} and type {}:".format(len(train_data), type(train_data)))
        print("Valid_set data size {} and type {}: ".format(len(valid_data), type(valid_data)))
        print("test_set data size {} and type {}: ".format(len(test_data), type(test_data)))


        # train partition
        subsets = {
            cid: Subset(train_data, train_targets,
                        partitioner.client_dict[cid],
                        transform=transform,
                        target_transform=target_transform)
            for cid in range(self.num_clients)
        }
        for cid in subsets:
            torch.save(
                subsets[cid],
                os.path.join(self.path, "train", "data{}.pkl".format(cid)))

        # val partition
        subsets = {
            cid: Subset(valid_data, valid_targets,
                        partitioner_val.client_dict[cid],
                        transform=transform,
                        target_transform=target_transform)
            for cid in range(self.num_clients)
        }
        for cid in subsets:
            torch.save(
                subsets[cid],
                os.path.join(self.path, "val", "data{}.pkl".format(cid)))

        # test partition
        subsets = {
            cid: Subset(testset.data, testset.targets,
                        partitioner_test.client_dict[cid],
                        transform=transform,
                        target_transform=target_transform)
            for cid in range(self.num_clients)
        }
        for cid in subsets:
            torch.save(
                subsets[cid],
                os.path.join(self.path, "test", "data{}.pkl".format(cid)))


    def get_dataset(self, cid, type="train"):
        """Load subdataset for client with client ID ``cid`` from local file.

        Args:
             cid (int): client id
             type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

        Returns:
            Dataset
        """
        dataset = torch.load(
            os.path.join(self.path, type, "data{}.pkl".format(cid)))
        return dataset

    def get_dataloader(self, cid, batch_size=None, type="train"):
        """Return dataload for client with client ID ``cid``.

        Args:
            cid (int): client id
            batch_size (int, optional): batch size in DataLoader.
            type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
        """
        dataset = self.get_dataset(cid, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader