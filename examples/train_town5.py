from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.model_se import Net
from ignite.metrics import Precision, Recall


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #print("Target is",target)
        #print("Model output is",output)
        #torch.nn.functional.nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        #input – (N,C) where C = number of classes or (N, C, H, W)(N,C,H,W) in case of 2D Loss...
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    precision = Precision()
    recall=Recall()
    with torch.no_grad():
        for data, target in test_loader:
            #print("Data is",data)
            data, target = data.to(device), target.to(device)
            
            print(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            precision.update((pred, target))
            recall.update((pred, target))
            #print("Pred",pred)
            #View this tensor as the same size as other. self.view_as(other) is equivalent to self.view(other.size()).
            # tensor.eq Computes element-wise equality
            #print("Sum is",pred.eq(target.view_as(pred)).sum())
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    try:
        print('\nTest set: Precision',precision.compute())
    except:
        pass
    print('\nTest set: Recall',recall.compute())

def sample_loader(path):
    arr=np.load(path)
    new_arr=np.expand_dims(arr,axis=0)
    return new_arr

def main():
      #training settings
      parser = argparse.ArgumentParser(description='Training Settings')
      parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
      parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 16)')
      parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
      parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
      parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
      parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
      parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
      parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
      parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
      parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
      args = parser.parse_args()
      
      device = torch.device("cuda")
      #A torch.device is an object representing the device on which a torch.Tensor is or will be allocated.
      
      transform=transforms.Compose([transforms.ToTensor()])
                                    
      train_data=datasets.DatasetFolder("training_data_town5",transform=transform,extensions=".npy",loader=sample_loader)
      train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=4,
                                          shuffle=True)

      test_data=datasets.DatasetFolder("test_data_town5",transform=transform,extensions=".npy",loader=sample_loader)
      test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=4,
                                          shuffle=True)
      
      model=Net().to(device)
      print(model)
      
      #To use torch.optim you have to construct an optimizer object, that will hold the current state 
      #and will update the parameters based on the computed gradients.
      optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
      scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
      

      for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()
      #state_dict(),returns a dictionary containing a whole state of the model
      torch.save(model.state_dict(), "safe_estimation_town5.pt")

if __name__ == '__main__':
    main()