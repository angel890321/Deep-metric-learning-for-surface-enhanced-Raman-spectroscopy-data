from torch.autograd import Variable
import torch
from pytorchtools import EarlyStopping
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
)
class Solver:
    def __init__(
        self,
        i,
        dataloaders,
        model,
        model_path,
        device,
        n_classes,
        criterion,
        criterion_proxy,
        optimizer,
        is_ce,
        gpu=-1,
    ):
        self.i = i
        self.dataloaders = dataloaders
        self.device = device
        self.net = model
        self.net = self.net.to(self.device)
        self.n_classes = n_classes
        self.criterion = criterion
        self.criterion_proxy = criterion_proxy
        self.optimizer = optimizer
        self.model_path = model_path
        self.is_ce = is_ce


    def iterate(self, epoch, phase):
        if phase == "train":
            self.net.train()
        else:
            self.net.eval()

            
        dataloader = self.dataloaders[phase]
        total_loss = 0
        correct = 0
        total = 0
            
        for batch_idx, (inputs, targets,patients) in enumerate(dataloader):
            inputs, targets,patients = Variable(inputs).to(self.device), Variable(targets.long()).to(self.device), patients
            out,emb= self.net(inputs)

            if self.is_ce == True:
            
                loss = self.criterion(out, targets)

            else:
                loss1 = self.criterion(out, targets)
                loss2 = self.criterion_proxy(emb, targets)
                loss = loss1 + 0.5 * loss2

            
            outputs = torch.argmax(out.detach(), dim=1)
            
            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.is_ce == False:
                    torch.nn.utils.clip_grad_value_(self.net.parameters(), 10)
                    torch.nn.utils.clip_grad_value_(self.criterion_proxy.parameters(), 10)


            total_loss += loss.item()
            total += targets.size(0)
            correct += outputs.eq(targets).cpu().sum().item()

        total_loss /= (batch_idx + 1)
        total_acc = correct / total

                
        #print("\nepoch %d: %s average loss: %.3f | acc: %.2f%% (%d/%d)"
                #% (epoch + 1, phase, total_loss, 100.0 * total_acc, correct, total))

        return total_loss,total_acc,out,targets
            

    def train(self, epochs):
        best_loss = float("inf")
        best_acc = 0
        epoch_breaks_classifer = 0
        train_losses = []
        train_acces = [] 
        test_losses = []
        test_acces = [] 


        early_stopping_classifier = EarlyStopping(patience=25, mode='acc', verbose=True)

        for epoch in tqdm(range(epochs)):
            train_loss,train_acc,out,targets =self.iterate(epoch, "train")

            train_losses.append(train_loss)
            train_acces.append(train_acc)

            with torch.no_grad():
                
                test_loss,test_acc,out,targets= self.iterate(epoch, "test")
                    
                test_losses.append(test_loss)
                test_acces.append(test_acc)

                if test_acc > best_acc:
                    best_acc = test_acc
                    checkpoint = {
                        "epoch": epoch,
                         "train_acc":train_acc,
                        "test_acc":test_acc,
                        "test_loss": test_loss,
                        "state_dict": self.net.state_dict(),
                    }
                    #print("best test acc found")
                    torch.save(checkpoint, self.model_path)

                early_stopping_classifier(test_acc)

                epoch_breaks_classifer+= 1

                if early_stopping_classifier.early_stop:
                    break  # 停止訓練
        
        if not early_stopping_classifier.early_stop:
            epoch_breaks_classifer = epochs
        
    
    def test_iterate(self, phase):
        self.net.eval()
        dataloader = self.dataloaders[phase]
        y_pred = []
        y_true = []
        y_pred_prob = []
        
        # 儲存模型的 embedding feature 以繪製tsne圖
        features_out = []
        features_combine = []
        targets_combine = []
        patient_ids = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets,patients) in enumerate(dataloader):
                inputs, targets ,patients = Variable(inputs).to('cuda'), Variable(targets.long()).to('cuda'), patients
                out,emb = self.net(inputs)
                outputs = torch.argmax(out.detach(), dim=1)
                outputs_prob = nn.functional.softmax(out.detach(), dim=1)
                y_pred.append(outputs.cpu().numpy())
                y_pred_prob.append(outputs_prob.cpu().numpy())
            
                y_true.append(targets.cpu().numpy())
                
                targets_combine.append(targets.detach().cpu().numpy())
                patient_ids.append(patients)
                features_out.append(out.cpu().numpy())
                features_combine.append(emb.detach().cpu().numpy())

            targets_combine = np.concatenate(targets_combine, axis=0) 
            patient_ids = np.concatenate( patient_ids, axis=0) 
            features_out = np.concatenate(features_out, axis=0)
            features_combine = np.concatenate(features_combine, axis=0)
        
        return (
                np.hstack(y_pred),
                np.hstack(y_true),
                np.vstack(y_pred_prob),
                features_out,
                features_combine,
                targets_combine,
                patient_ids,
            )
    def test(self):
        checkpoint = torch.load(self.model_path)
        epoch = checkpoint["epoch"]
        train_acc = checkpoint["train_acc"]
        test_acc = checkpoint["test_acc"]
        self.net.load_state_dict(checkpoint["state_dict"])
        #print("load model at epoch {}, with test acc : {:.3f}".format(epoch+1, test_acc))

        
        _, _ ,_,train_out,train_combined,train_targets,train_patient_ids = self.test_iterate("train")
        y_pred, y_true ,y_pred_prob,out,combined,targets, test_patient_ids,= self.test_iterate("test")
        
        #print("total", accuracy_score(y_true, y_pred))
        #for i in range(self.n_classes):
            #idx = y_true == i
            #print("class", i, accuracy_score(y_true[idx], y_pred[idx]))

        return (
            confusion_matrix(y_true, y_pred),
            y_true,
            y_pred,
            accuracy_score(y_true, y_pred),
            y_pred_prob,
            train_acc,
            train_targets,
            targets,
            train_combined,
            combined,
            train_patient_ids,
            test_patient_ids
        )