import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from options import args_parser
from collections import defaultdict

def get_dataset(dataset):
    if dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        samples_per_label = 250
        
    elif dataset == 'MNIST':
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        samples_per_label = 300
        
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        dataset_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        samples_per_label = 300

    elif dataset == 'PACS':
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        domains = ['sketch', 'cartoon', 'art_painting', 'photo']
        dataset_train = {}
        dataset_test_domain = {}
        test_datasets = []
        
        for i, domain in enumerate(domains):
            full_dataset = datasets.ImageFolder(root=f"data/PACS/{domain}", transform=transform)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
            train_dataset = CustomSubset(full_dataset, train_dataset.indices)
            test_dataset = CustomSubset(full_dataset, test_dataset.indices)
            dataset_train[i] = train_dataset
            dataset_test_domain[i] = test_dataset
            test_datasets.append(test_dataset)

        dataset_test = ConcatDataset(test_datasets)
        return dataset_train, dataset_test, dataset_test_domain

    return dataset_train, dataset_test, samples_per_label


class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.classes = dataset.classes
        
def check_for_image_reuse(client_data_train):
    image_usage_counts = {}
    for client_id in client_data_train:
        for time_id in range(len(client_data_train[client_id])):
            data, labels = client_data_train[client_id][time_id]
            indices = torch.arange(len(labels))
            for index in indices:
                image_id = labels[index].item()
                if image_id not in image_usage_counts:
                    image_usage_counts[image_id] = 0
                image_usage_counts[image_id] += 1
    reused_images = {image_id: count for image_id, count in image_usage_counts.items() if count > 1}
    if reused_images:
        print("Some images are reused among clients:")
        for image_id, count in reused_images.items():
            print(f"Image ID {image_id} is used {count} times.")
    else:
        print("No image reuse detected. All images are uniquely assigned.")


def print_class_distribution(datasets_train):
    for domain, dataset in datasets_train.items():
        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        _, targets = next(iter(data_loader))
        class_counts = defaultdict(int)
        for label in targets:
            class_counts[label.item()] += 1
        print(f"\nDomain: {domain}")
        class_idx_count = ''
        for class_idx, count in sorted(class_counts.items()):
            class_idx_count = class_idx_count + f"Class {class_idx}: {count}, "
        print(class_idx_count)

def Domain_IID(datasets_train, dataset_test, dataset_test_domain, num_clients=10):
    domains = list(datasets_train.keys())
    client_data_train = {i: [] for i in range(num_clients)}
    used_indices = {domain: set() for domain in domains}

    for domain in domains:
        data_loader = DataLoader(datasets_train[domain], batch_size=len(datasets_train[domain]), shuffle=True)
        x_train, y_train = next(iter(data_loader))
        class_indices = {i: (y_train == i).nonzero(as_tuple=True)[0] for i in range(len(datasets_train[domain].classes))}
        samples_per_client_per_class = {i: len(indices) // num_clients for i, indices in class_indices.items()}
        
        for client_id in range(num_clients):
            chosen_indices = []
            for class_id in range(len(datasets_train[domain].classes)):
                available_indices = [idx for idx in class_indices[class_id] if idx not in used_indices[domain]]
                num_samples = min(samples_per_client_per_class[class_id], len(available_indices))
                selected_indices = np.random.choice(available_indices, num_samples, replace=False)
                used_indices[domain].update(selected_indices)
                chosen_indices.extend(selected_indices)
            
            client_data_train[client_id].append((x_train[chosen_indices], y_train[chosen_indices]))

    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
    client_data_test = next(iter(test_loader))
    client_data_test_domain = {}
    for domain, dataset in dataset_test_domain.items():
        test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        client_data_test_domain[domain] = next(iter(test_loader))
    
    for client_id in range(3):
        print([f"Time ID {time_id}: Labels {torch.unique(labels, return_counts=True)}" for time_id, (data, labels) in enumerate(client_data_train[client_id])])

    for i in range(4):
        total_samples = sum(len(client_data_train[client][i][0]) for client in client_data_train)
        print(f"Total samples across all clients and time_ids: {total_samples}")
    return client_data_train, client_data_test, client_data_test_domain

def Domain_NonIID(datasets_train, dataset_test, num_clients=10):
    domains = list(datasets_train.keys())
    client_data_train = {i: [] for i in range(num_clients)}
    used_indices = {domain: set() for domain in domains}

    def get_class_ids(client_id, num_classes):
        return [(client_id * 2 + i) % num_classes for i in range(2)]

    for domain in domains:
        data_loader = DataLoader(datasets_train[domain], batch_size=len(datasets_train[domain]), shuffle=True)
        x_train, y_train = next(iter(data_loader))
        class_indices = {i: (y_train == i).nonzero(as_tuple=True)[0] for i in range(len(datasets_train[domain].classes))}

        min_samples_per_class = min(len(indices) for indices in class_indices.values()) // num_clients
        min_samples_per_client = min_samples_per_class * 2 

        for client_id in range(num_clients):
            class_ids = get_class_ids(client_id, len(datasets_train[domain].classes))
            chosen_indices = []
            for class_id in class_ids:
                available_indices = [idx for idx in class_indices[class_id] if idx not in used_indices[domain]]
                selected_indices = np.random.choice(available_indices, min_samples_per_class, replace=False)
                used_indices[domain].update(selected_indices)
                chosen_indices.extend(selected_indices)
            
            client_data_train[client_id].append((x_train[chosen_indices], y_train[chosen_indices]))

    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
    client_data_test = next(iter(test_loader))
    for client_id in range(8):
        print([f"Time ID {time_id}: Labels {torch.unique(labels, return_counts=True)}" for time_id, (data, labels) in enumerate(client_data_train[client_id])])

    for i in range(3):
        total_samples = sum(len(client_data_train[client][i][0]) for client in client_data_train)
        print(f"Total samples across all clients and time_ids: {total_samples}")

    check_for_image_reuse(client_data_train)
    return client_data_train, client_data_test

def Class_IID(dataset_train, dataset_test, samples_per_label, num_clients=20, original_clients=100):
    samples_per_label = int(60000 / 5 / num_clients / 2)
    original_clients = int(num_clients * 5)
    train_loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
    x_train, y_train = next(iter(train_loader)); x_test, y_test = next(iter(test_loader))

    idx_label = {i: [] for i in range(10)}
    for idx, label in enumerate(y_train):
        idx_label[label.item()].append(idx)

    client_data_train = {i: [] for i in range(num_clients)}
    for i in range(original_clients):
        chosen_labels = [(i*2)%10, (i*2+1)%10]
        chosen_indices = []
        for label in chosen_labels:
            if len(idx_label[label]) < samples_per_label:
                idx_label[label] = list(torch.where(y_train == label)[0].cpu().numpy())
            label_indices = idx_label[label][:samples_per_label]
            chosen_indices.extend(label_indices)
            idx_label[label] = idx_label[label][samples_per_label:]
        client_indices_tensor = torch.tensor(chosen_indices, dtype=torch.long)
        client_id = i // (original_clients // num_clients)
        if not client_data_train[client_id]:
            for _ in range(5):
                client_data_train[client_id].append(([], []))
        client_data_train[client_id][i % 5][0].append(x_train[client_indices_tensor])
        client_data_train[client_id][i % 5][1].append(y_train[client_indices_tensor])

    for client_id in range(num_clients):
        for time_id in range(5):
            client_data_train[client_id][time_id] = (
                torch.cat(client_data_train[client_id][time_id][0], dim=0),
                torch.cat(client_data_train[client_id][time_id][1], dim=0))

    client_data_test = (x_test, y_test)
    print("Client train labels (for time slices of Client 0):", [torch.unique(client_data_train[0][i][1]) for i in range(5)])
    print("Client number of samples:", (client_data_train[0][0][1] == 0).sum().item(), (client_data_train[0][0][1] == 1).sum().item())
    print("Client train labels (for time slices of Client 1):", [torch.unique(client_data_train[1][i][1]) for i in range(5)])
    return client_data_train, client_data_test

def Class_NonIID(dataset_train, dataset_test, samples_per_label, num_clients=20, original_clients=100):
    train_loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
    x_train, y_train = next(iter(train_loader)); x_test, y_test = next(iter(test_loader))

    idx_label = {i: [] for i in range(10)}
    for idx, label in enumerate(y_train):
        idx_label[label.item()].append(idx)

    client_data_train = {i: [] for i in range(num_clients)}
    for i in range(original_clients):
        chosen_labels = [(i*2)%10, (i*2+1)%10]
        chosen_indices = []
        for label in chosen_labels:
            if len(idx_label[label]) < samples_per_label:
                idx_label[label] = list(torch.where(y_train == label)[0].cpu().numpy())
            label_indices = idx_label[label][:samples_per_label]
            chosen_indices.extend(label_indices)
            idx_label[label] = idx_label[label][samples_per_label:]
        client_indices_tensor = torch.tensor(chosen_indices, dtype=torch.long)
        client_id = i // (original_clients // num_clients)
        if not client_data_train[client_id]:
            for _ in range(5):
                client_data_train[client_id].append(([], []))
        client_data_train[client_id][i % 5][0].append(x_train[client_indices_tensor])
        client_data_train[client_id][i % 5][1].append(y_train[client_indices_tensor])

    for client_id in range(num_clients):
        for time_id in range(5):
            client_data_train[client_id][time_id] = (
                torch.cat(client_data_train[client_id][time_id][0], dim=0),
                torch.cat(client_data_train[client_id][time_id][1], dim=0))
    
    for client in client_data_train:
        ite_list = client_data_train[client]
        client_data_train[client] = ite_list[client % len(ite_list):] + ite_list[:client % len(ite_list)]
    client_data_test = (x_test, y_test)
    print("Client train labels (for time slices of Client 0):", [torch.unique(client_data_train[0][i][1]) for i in range(5)])
    print("Client number of samples:", (client_data_train[0][0][1] == 0).sum().item(), (client_data_train[0][0][1] == 1).sum().item())
    print("Client train labels (for time slices of Client 1):", [torch.unique(client_data_train[1][i][1]) for i in range(5)])
    return client_data_train, client_data_test

