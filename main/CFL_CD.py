import numpy as np                    
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.optim as optim
from torch.nn import DataParallel
import copy

from options import args_parser
from load_dataset import get_dataset, Domain_NonIID, Domain_IID, Class_IID, Class_NonIID
from models import MNIST_CNN, CIFAR10_CNN, PACS_CNN, ConditionalUNet, DenoiseDiffusion

class CFL_CD():
    def __init__(self, args, model, client_data_train, client_data_test, diffusion_models, iterations, sessions, local_epochs):
        self.args = copy.deepcopy(args)
        self.model = model
        self.criterion = nn.NLLLoss()
        self.client_data_train = client_data_train
        self.client_data_test = client_data_test
        self.diffusion_models = diffusion_models
        self.appeared_classes = []
        self.iterations = iterations
        self.sessions = sessions
        self.local_epochs = local_epochs
        
    def average_weights(self, weights_list):
        avg_weights = {}
        for key in weights_list[0].keys():
            if weights_list[0][key].dtype != torch.long:
                avg_weights[key] = torch.mean(torch.stack([weights[key] for weights in weights_list]), dim=0)
        return avg_weights
    
    def train_client(self, client_data, global_model):
        client_model = type(self.model)().to(self.args.device)
        client_model.load_state_dict(copy.deepcopy(global_model).state_dict())
        optimizer = optim.Adam(client_model.parameters(), lr=0.0001)
        data, target = client_data
        data = data.float().to(self.args.device); target = target.long().to(self.args.device)
        data_loader = torch.utils.data.DataLoader(list(zip(data, target)), batch_size=32, shuffle=True)
        for epoch in range(self.local_epochs):
            total_loss = 0
            correct = 0
            total = 0
            for batch_data, batch_target in data_loader:
                optimizer.zero_grad()
                output = client_model(batch_data)
                loss = self.criterion(output, batch_target)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
        
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(batch_target.view_as(pred)).sum().item()
                total += batch_target.size(0)
            
            accuracy = 100. * correct / total
            avg_loss = total_loss / len(data_loader)
        return client_model, avg_loss, accuracy
    
    def test_client(self, client_data, global_model):
        global_model = global_model.to(self.args.device)
        data, target = client_data
        data = data.float().to(self.args.device); target = target.long().to(self.args.device)
        with torch.no_grad():
            output = global_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / target.size(0)
        return accuracy

    def test_client_classIID(self, client_data, global_model):
        global_model = global_model.to(self.args.device)
        data, target = client_data
        mask = np.isin(target, list(self.appeared_classes))
        data = data[mask].float().to(self.args.device)
        target = target[mask].long().to(self.args.device)
    
        with torch.no_grad():
            output = global_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / target.size(0)
        return accuracy
    
    def train_diffusion_model(self, diffusion_model, client_data, epochs=1000):
        data, target = client_data
        data = data.float().to(self.args.device); target = target.float().to(self.args.device)
        data_loader = torch.utils.data.DataLoader(list(zip(data, target)), batch_size=32, shuffle=True)
        optimizer = optim.Adam(diffusion_model.eps_model.parameters(), lr=0.0001, weight_decay=1e-5)
        for epoch in range(epochs):
            for batch_data, batch_target in data_loader:
                optimizer.zero_grad()
                loss = diffusion_model.loss(batch_data, batch_target)
                loss.backward()
                optimizer.step()
        return diffusion_model

    def generate_synthetic_data(self, diffusion_model, amount, condition_labels):
        # amount = args.synthetic_number
        image_shape = self.args.image_shape
        synthetic_data_list = []; synthetic_labels_list = []
        amount_per_label = amount // len(condition_labels)
        with torch.no_grad():
            for label in condition_labels:
                data = torch.randn((amount_per_label, *image_shape), device=self.args.device)
                labels = torch.full((amount_per_label, 1), label, dtype=torch.float, device=self.args.device)
                for t_ in range(diffusion_model.n_steps):
                    t = diffusion_model.n_steps - t_ - 1
                    data = diffusion_model.p_sample(data, data.new_full((amount_per_label,), t, dtype=torch.long), labels)
                synthetic_labels_list.append(labels.squeeze())
                synthetic_data_list.append(data)
        synthetic_data = torch.cat(synthetic_data_list)
        synthetic_labels = torch.cat(synthetic_labels_list)
        return synthetic_data.cpu(), synthetic_labels.squeeze().cpu()

    def generate_synthetic_data_domian(self, session, diffusion_model, amount, condition_labels):
        image_shape = self.args.image_shape
        synthetic_data_list = []
        synthetic_labels_list = {"label": [], "session": []}
        amount_per_label_per_session = amount // len(condition_labels) // (session + 1)
        with torch.no_grad():
            for label in condition_labels:
                for sess in range(session + 1):
                    data = torch.randn((amount_per_label_per_session, *image_shape), device=self.args.device)
                    labels = torch.full((amount_per_label_per_session, 1), label, dtype=torch.float, device=self.args.device)
                    session_tensor = torch.full((amount_per_label_per_session, 1), sess, dtype=torch.float, device=self.args.device)
                    combined_labels = torch.cat((labels, session_tensor), dim=1)
                    for t_ in range(diffusion_model.n_steps):
                        t = diffusion_model.n_steps - t_ - 1
                        data = diffusion_model.p_sample(data, data.new_full((amount_per_label_per_session,), t, dtype=torch.long), combined_labels)
                    synthetic_data_list.append(data)
                    synthetic_labels_list["label"].append(labels)
                    synthetic_labels_list["session"].append(session_tensor)
        synthetic_data = torch.cat(synthetic_data_list)
        synthetic_labels = {key: torch.cat(value).cpu() for key, value in synthetic_labels_list.items()}
        return synthetic_data.cpu(), synthetic_labels
    
    def combine_data(self, real_data, synthetic_dataset):
        real_x, real_y = real_data
        synthetic_data, synthetic_label = synthetic_dataset
        real_x = real_x.float(); real_y = real_y.long()
        combined_x = torch.cat([real_x, synthetic_data], dim=0)
        combined_y = torch.cat([real_y, synthetic_label], dim=0)
        return combined_x, combined_y
        
    def Diffusion(self):
        global_model = self.model
        accuracy_matrix = []
        accuracy_matrix_ClassIID = []
        accuracy_matrix_DomainIID = []
        diffusion_models = copy.deepcopy(self.diffusion_models)
        
        for ite in range(self.iterations):
            session = (ite // (self.iterations//self.sessions)) % self.sessions
            if ite % (self.iterations//self.sessions) == (self.iterations//self.sessions)-1:
                for client in self.client_data_train:
                    if ite >= (self.iterations//self.sessions): 
                        client_dataset = [torch.cat((client_data_train[client][session][i], synthetic_dataset[client][i]), dim=0) for i in range(2)]
                        diffusion_models[client] = self.train_diffusion_model(copy.deepcopy(diffusion_models[client]), client_dataset)      
                    else:
                        diffusion_models[client] = self.train_diffusion_model(copy.deepcopy(diffusion_models[client]), client_data_train[client][session])
                    
            if ite % (self.iterations//self.sessions) == 0:
                combined_client_data = {} 
                synthetic_dataset = {}
                for client in self.client_data_train:
                    if ite >= (self.iterations//self.sessions): 
                        appeared_labels = torch.cat([torch.unique(client_data_train[client][i][1]) for i in range(session)], dim=0)
                        synthetic_data, synthetic_label = self.generate_synthetic_data(diffusion_models[client], amount=len(client_data_train[client][session][0]), condition_labels=appeared_labels)
                        synthetic_dataset[client] = [synthetic_data, synthetic_label]
                        combined_client_data[client] = self.combine_data(client_data_train[client][session], synthetic_dataset[client])
                        
                    else:
                        combined_client_data[client] = client_data_train[client][session]
    
            client_models = []
            for client in self.client_data_train:
                client_model, avg_loss, accuracy = self.train_client(combined_client_data[client], global_model)
                client_models.append(client_model.state_dict())
    
            global_weights = self.average_weights(client_models)
            global_model.load_state_dict(global_weights)
            if args.task == 'Class_IID':
                self.appeared_classes.append(torch.unique(self.client_data_train[client][session][1]).numpy())
                test_accuracy_ClassIID = self.test_client_classIID(self.client_data_test, global_model)
                print(f"Iteration: {ite} - Appeared Test Acc: {test_accuracy_ClassIID:.2f}%")
                accuracy_matrix_ClassIID.append(test_accuracy_ClassIID)
                file_name = f'/home/liangqiy/Desktop/CFL_Diffusion/{args.dataset}/{args.task}_CFL_{args.dataset}_{args.framework}_NumClients_{args.client_number}_ClassIID.npy'
                np.save(file_name, accuracy_matrix_ClassIID)
                
            test_accuracy = self.test_client(self.client_data_test, global_model)
            accuracy_matrix.append(test_accuracy)
            print(f"Iteration: {ite} - Test Acc: {test_accuracy:.2f}%")
            CNN_file_name = f'/home/liangqiy/Desktop/CFL_Diffusion/{args.dataset}/model/CNN_{args.task}_CFL_{args.dataset}_{args.framework}.pth'
            Diffusion_file_name = f'/home/liangqiy/Desktop/CFL_Diffusion/{args.dataset}/model/Diffusion_{args.task}_CFL_{args.dataset}_{args.framework}.pth'
            torch.save(global_model.state_dict(), CNN_file_name)
            torch.save(diffusion_models[0].eps_model.state_dict(), Diffusion_file_name)
        return accuracy_matrix

    def Domain_Diffusion(self):
        global_model = self.model
        accuracy_matrix = []
        accuracy_matrix_ClassIID = []
        accuracy_matrix_DomainIID = []
        diffusion_models = copy.deepcopy(self.diffusion_models)
        
        for ite in range(self.iterations):
            session = (ite // (self.iterations//self.sessions)) % self.sessions
            if ite % (self.iterations//self.sessions) == (self.iterations//self.sessions)-1 and ite != self.iterations-1:
                for client in self.client_data_train:
                    if ite >= (self.iterations//self.sessions): 
                        train_data, train_labels = client_data_train[client][session]
                        synthetic_data, synthetic_labels = synthetic_dataset[client]
                        session_tensor = torch.full_like(train_labels, session, dtype=torch.float)
                        extended_train_labels = torch.cat((train_labels.unsqueeze(1), session_tensor.unsqueeze(1)), dim=1)
                        combined_data = torch.cat((train_data, synthetic_data), dim=0)
                        synthetic_combined_labels = torch.cat((synthetic_labels['label'], synthetic_labels['session']), dim=1)
                        combined_labels = torch.cat((extended_train_labels, synthetic_combined_labels), dim=0)
                        client_dataset = (combined_data, combined_labels)
                        diffusion_models[client] = self.train_diffusion_model(copy.deepcopy(diffusion_models[client]), client_dataset)      
                    else:
                        train_data, train_labels = client_data_train[client][session]
                        session_tensor = torch.full_like(train_labels, session, dtype=torch.float)
                        extended_train_labels = torch.cat((train_labels.unsqueeze(1), session_tensor.unsqueeze(1)), dim=1)
                        client_dataset = (train_data, extended_train_labels)
                        diffusion_models[client] = self.train_diffusion_model(copy.deepcopy(diffusion_models[client]), client_dataset)
                    
            if ite % (self.iterations//self.sessions) == 0:
                combined_client_data = {} 
                synthetic_dataset = {}
                for client in self.client_data_train:
                    if ite >= (self.iterations//self.sessions): 
                        appeared_labels = torch.cat([torch.unique(client_data_train[client][i][1]) for i in range(session)], dim=0)
                        synthetic_data, synthetic_label = self.generate_synthetic_data_domian(session, diffusion_models[client], amount=len(client_data_train[client][session][0]), condition_labels=appeared_labels)
                        synthetic_dataset[client] = [synthetic_data, synthetic_label]
                        client_data = [synthetic_data, synthetic_label["label"].squeeze(1)]
                        combined_client_data[client] = self.combine_data(client_data_train[client][session], client_data)
                        
                    else:
                        combined_client_data[client] = client_data_train[client][session]
    
            client_models = []
            for client in self.client_data_train:
                client_model, avg_loss, accuracy = self.train_client(combined_client_data[client], global_model)
                client_models.append(client_model.state_dict())
    
            global_weights = self.average_weights(client_models)
            global_model.load_state_dict(global_weights)

            if session >= 1:
                data_test_appeared_domains = (torch.cat([client_data_test_domain[domain][0] for domain in range(session + 1)], dim=0), 
                                              torch.cat([client_data_test_domain[domain][1] for domain in range(session + 1)], dim=0))
            else:
                data_test_appeared_domains = client_data_test_domain[0]
            test_accuracy_Domain_IID = self.test_client(data_test_appeared_domains, global_model)
            print(f"Iteration: {ite} - Appeared Test Acc: {test_accuracy_Domain_IID:.2f}%")
            accuracy_matrix_DomainIID.append(test_accuracy_Domain_IID)
            file_name = f'/home/liangqiy/Desktop/CFL_Diffusion/{args.dataset}/{args.task}_CFL_{args.dataset}_{args.framework}_DomainIID.npy'
            np.save(file_name, accuracy_matrix_DomainIID)
                
            test_accuracy = self.test_client(self.client_data_test, global_model)
            accuracy_matrix.append(test_accuracy)
            print(f"Iteration: {ite} - Test Acc: {test_accuracy:.2f}%")
            CNN_file_name = f'/home/liangqiy/Desktop/CFL_Diffusion/{args.dataset}/model/CNN_{args.task}_CFL_{args.dataset}_{args.framework}.pth'
            Diffusion_file_name = f'/home/liangqiy/Desktop/CFL_Diffusion/{args.dataset}/model/Diffusion_{args.task}_CFL_{args.dataset}_{args.framework}.pth'
            torch.save(global_model.state_dict(), CNN_file_name)
            torch.save(diffusion_models[0].eps_model.state_dict(), Diffusion_file_name)
        return accuracy_matrix


if __name__ == '__main__':
    args = args_parser()
    dataset_train, dataset_test, samples_per_label = get_dataset(dataset=args.dataset)
    
    if args.task == 'Class_NonIID':
        client_data_train, client_data_test = Class_NonIID(dataset_train, dataset_test, samples_per_label, num_clients=20, original_clients=100)
    elif args.task == 'Class_IID': 
        client_data_train, client_data_test = Class_IID(dataset_train, dataset_test, samples_per_label, num_clients=20, original_clients=100)
    elif args.task == 'Domain_NonIID': 
        client_data_train, client_data_test = Domain_NonIID(dataset_train, dataset_test, num_clients=10)
    elif args.task == '4Domain_IID_C2': 
        client_data_train, client_data_test, client_data_test_domain = Domain_IID(dataset_train, dataset_test, samples_per_label, num_clients=10)

    args.image_shape = client_data_train[0][0][0][0].shape
    iterations = 100
    sessions=5

    if args.dataset == 'MNIST':
        model = MNIST_CNN()
        eps_model = ConditionalUNet(image_channels=1, n_channels=64, condition_dim=32, ch_mults=(1, 2, 2, 4), is_attn=[False, False, False, True])
    elif args.dataset == 'FashionMNIST':
        model = MNIST_CNN()
        eps_model = ConditionalUNet(image_channels=1, n_channels=64, condition_dim=32, ch_mults=(1, 2, 2, 4), is_attn=[False, False, False, True])
    elif args.dataset == 'CIFAR10':
        model = CIFAR10_CNN()
        eps_model = ConditionalUNet(image_channels=3, n_channels=64, condition_dim=32, ch_mults=(1, 2, 2, 4), is_attn=[False, False, False, True])
    elif args.dataset == 'PACS':
        model = PACS_CNN()
        eps_model = ConditionalUNet(image_channels=3, n_channels=64, condition_dim=32, ch_mults=(1, 2, 2, 4), is_attn=[False, False, False, True])
        eps_model.time_emb.lin1 = nn.Linear(eps_model.time_emb.n_channels // 4 + 2, eps_model.time_emb.lin1.out_features)
        iterations = 80
        sessions=4
        
    diffusion_models = {}
    for client_id in range(len(client_data_train)):
        eps_model = eps_model.to(args.device)
        diffusion_models[client_id] = DenoiseDiffusion(eps_model=eps_model, n_steps=1000, device=args.device)
    
    CFL_CD = CFL_CD(args, model, client_data_train, client_data_test, diffusion_models, iterations=iterations, sessions=sessions, local_epochs=5)
    if args.task == '4Domain_IID_C2': 
       accuracy_matrix = CFL_CD.Domain_Diffusion() 
    else:
        accuracy_matrix = CFL_CD.Diffusion()
        
    file_name = f'/home/liangqiy/Desktop/CFL_Diffusion/{args.dataset}/{args.task}_CFL_{args.dataset}_{args.framework}.npy'
    np.save(file_name, accuracy_matrix)


