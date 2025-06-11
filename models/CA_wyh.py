import os
import pandas as pd
import numpy as np
import collections
import sys
import torch.nn.functional as F
from .modelBase import modelBase
from utils import CHARAS_LIST
import pickle as pkl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt
MAX_EPOCH = 30


class CA_base_wyh(nn.Module,modelBase):
    def __init__(self,name,omit_char=[],device='cuda'):
        nn.Module.__init__(self)
        modelBase.__init__(self,name)
        self.ca_beta=None
        self.ca_factor=None
        self.train_loader=None
        self.valid_loader=None
        self.test_loader=None
        self.optimizer=None
        self.criterion=None
        self.factor_nn_pred=[]
        self.name=name
        self.device=device
        self.omit_char=omit_char
    def forward(self,beta_input,factor_input):
        beta_output=self.ca_beta(beta_input)
        factor_output=self.ca_factor(factor_input)
        # print(beta_output.shape,factor_output.shape)
        output=torch.sum(beta_output*factor_output,dim=1)
        return output
    
    
    
    def _get_dataset(self,period):
        with open('/home/wyh/DLFDA/autoencoder_we/data/mon_list.pkl','rb') as file:
            month_list=pkl.load(file).to_list()
        if period=='train':
            month=[i for i in month_list if i>=self.train_period[0] and i<=self.train_period[1]]
        if period=='valid':
            month=[i for i in month_list if i>=self.valid_period[0] and i<=self.valid_period[1]]
        if period=='test':
            month=[i for i in month_list if i>=self.test_period[0] and i<=self.test_period[1]]    
        
        self.p_charas=pd.read_pickle('/home/wyh/DLFDA/autoencoder_we/data/p_charas.pkl')
        self.portfolio_ret=pd.read_pickle('/home/wyh/DLFDA/autoencoder_we/data/portfolio_ret.pkl')
        
        betas=[]
        factors=[]
        labels=[]
        for mon in month:
            beta=self.p_charas.loc[self.p_charas['DATE']==mon][CHARAS_LIST].T.values    #  94*94
            label=self.portfolio_ret.loc[self.portfolio_ret['DATE']==mon][CHARAS_LIST].T.values
            factor=self.portfolio_ret.loc[self.portfolio_ret['DATE']==mon][CHARAS_LIST].T.values   #94*94
            
            betas.append(beta)
            factors.append(factor)
            labels.append(label)
            
        betas_tensor=torch.tensor(betas,dtype=torch.float32).to(self.device)
        factors_tensor=torch.tensor(factors,dtype=torch.float32).to(self.device)    
        labels_tensor=torch.tensor(labels,dtype=torch.float32).to(self.device)
        dataset=TensorDataset(betas_tensor,factors_tensor,labels_tensor)
        
        return DataLoader(dataset,batch_size=1,shuffle=True)
        
        
    def train_one_epoch(self):
        
        self.train_loader=self._get_dataset(period='train')
        epoch_loss=0
        
        for i,(beta_nn_input,factor_nn_input,label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            beta_nn_input=beta_nn_input.squeeze(0).T
            factor_nn_input=factor_nn_input.squeeze(0).T
            label=label.squeeze(0)
            output = self.forward(beta_nn_input, factor_nn_input)
            # print('output and label',output.shape,label.shape)
            loss = self.criterion(output, label)
            
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                # print(f'Batches: {i}, loss: {loss.item()}')
                pass

        return epoch_loss / len(self.train_loader)
    
    def _valid_one_epoch(self):
        epoch_loss = 0.0
        for i, (beta_nn_input, factor_nn_input, labels) in enumerate(self.valid_loader):
            # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
            # labels reshape: (1, 94) -> (94, ) (1*N => N,)
            beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)

            output = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(output, labels)
            epoch_loss += loss.item()

        return epoch_loss / len(self.valid_loader)
    
    def train_model(self):
        if 'saved_models' not in os.listdir('./'):
            os.mkdir('saved_models')
        
        self.train_loader=self._get_dataset(period='train')
        self.valid_loader=self._get_dataset(period='valid')
        
        
        min_error=np.Inf
        train_loss=[]
        valid_loss=[]
        for epoch in range(MAX_EPOCH):
            self.train()
            trainloss=self.train_one_epoch()
            train_loss.append(trainloss)
            
            self.eval()
            with torch.no_grad():
                validloss=self._valid_one_epoch()    
                valid_loss.append(validloss)
            
            if validloss < min_error:
                min_error = validloss
                no_update_steps = 0
                # save model
                torch.save(self.state_dict(), f'./saved_models/{self.name}.pt')
            else:
                no_update_steps += 1
                
            # if no_update_steps > 2: # early stop, if consecutive 3 epoches no improvement on validation set
            #     print(f'Early stop at epoch {epoch}')
                # print("train_loss:",train_loss)
                # print("valid_loss:",valid_loss)
                # break
            # load from (best) saved model
            self.load_state_dict(torch.load(f'./saved_models/{self.name}.pt'))
        return train_loss, valid_loss
    
    def test_model(self):
        # beta, factor, label = self.test_dataset
        # i = np.random.randint(len(beta))
        # beta_nn_input = beta[i]
        # factor_nn_input = factor[i]
        # labels = label[i]
        self.test_loader=self._get_dataset(period='test')
        output = None
        label = None
        for i, beta_nn_input, factor_nn_input, labels in enumerate(self.test_loader):
            # convert to tensor
            # beta_nn_input = torch.tensor(beta_nn_input, dtype=torch.float32).T.to(self.device)
            # factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device)
            # labels = torch.tensor(labels, dtype=torch.float32).T.to(self.device)
            output = self.forward(beta_nn_input, factor_nn_input)
            break

        loss = self.criterion(output, labels)
        print(f'Test loss: {loss.item()}')
        print(f'Predicted: {output}')
        print(f'Ground truth: {labels}')
        return output, labels
    
    def calBeta(self,month,skip_char=[]):
        
        _,beta,_,_=self._get_item(month)
        
        if len(skip_char):
            beta = pd.DataFrame(beta.T, columns=CHARAS_LIST) # N*P
            beta[skip_char] = beta[skip_char] * 0.0
            beta = beta.values.T # P*N
        
        
        beta_tensor=torch.tensor(beta,dtype=torch.float32).T.to(self.device)
        if self.name[0]=='V':
            beta_encode=self.beta_encoder(beta_tensor)
            mu=self.mu(beta_encode)
            logvar=self.logvar(beta_encode)
            beta_reparam=self.reparameter(mu,logvar)
            # beta=self.beta_decoder(beta_reparam)
            beta=beta_reparam
        else:
            beta=self.ca_beta(beta_tensor)
        return beta
    
    def calFactor(self,month,skip_char=[]):
        _,_,factor,_=self._get_item(month)    
        if len(skip_char):
            factor = pd.DataFrame(factor.T, columns=CHARAS_LIST) # 1*P
            factor[skip_char] = factor[skip_char] * 0.0
            factor = factor.values.T # P*1
        factor_tensor=torch.tensor(factor,dtype=torch.float32).T.to(self.device)
        factor=self.ca_factor(factor_tensor).T
        self.factor_nn_pred.append(factor)
        return factor
    
    def inference(self, month):
        if len(self.omit_char) == 0:
            assert month >= self.test_period[0], f"Month error, {month} is not in test period {self.test_period}"
            mon_factor, mon_beta = self.calFactor(month), self.calBeta(month)
            assert mon_beta.shape[1] == mon_factor.shape[0], f"Dimension mismatch between mon_factor: {mon_factor.shape} and mon_beta: {mon_beta.shape}"
            
            # R_{N*1} = Beta_{N*K} @ F_{K*1}
            return mon_beta @ mon_factor
        else:
            ret_R = []
            for char in self.omit_char:
                mon_factor, mon_beta = self.calFactor(month, [char]), self.calBeta(month, [char])
                ret_R.append((mon_beta @ mon_factor).cpu().detach().numpy()) # N*1
                
            mon_factor, mon_beta = self.calFactor(month), self.calBeta(month)
            ret_R.append((mon_beta @ mon_factor).cpu().detach().numpy()) # also add complete result
            # print('inference',np.array(ret_R).squeeze(2).T,np.array(ret_R).squeeze(2).T.shape)
            return np.array(ret_R).squeeze(2).T # N*m
    
    
    def _get_item(self,month):
        betas=pd.read_pickle('/home/wyh/DLFDA/autoencoder_we/data/p_charas.pkl')
        factors=pd.read_pickle('/home/wyh/DLFDA/autoencoder_we/data/portfolio_ret.pkl')
        labels=pd.read_pickle('/home/wyh/DLFDA/autoencoder_we/data/portfolio_ret.pkl')
        beta_mon=betas.loc[betas['DATE']==month][CHARAS_LIST]
        factor_mon=factors.loc[factors['DATE']==month][CHARAS_LIST].T.values
        label_mon=labels.loc[labels['DATE']==month][CHARAS_LIST].T.values
        return beta_mon.index,beta_mon.T.values,factor_mon,label_mon
    
    def cal_delayed_Factor(self,month):
        if self.refit_cnt==0:
            delayed_factor=self.factor_nn_pred[0]
        else:
            delayed_factor=torch.mean(torch.stack(self.factor_nn_pred[:self.refit_cnt]),dim=0)
        return delayed_factor
    
    def reset_parameters(self):
        if self.name[0]=='V':
            for layer in self.beta_encoder:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            # for layer in self.beta_decoder:
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()
            # for layer in self.beta_decoder:
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()
        else:
            for layer in self.ca_beta:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for layer in self.ca_factor:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.optimizer.state=collections.defaultdict(dict)

    def release_gpu(self):
        if self.train_loader is not None:
            del self.train_loader
        if self.valid_loader is not None:
            del self.valid_loader
        if self.test_loader is not None:
            del self.test_loader
        torch.cuda.empty_cache()

class CA0_wyh(CA_base_wyh):
    def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,omit_char=[],device='cuda'):
        CA_base_wyh.__init__(self,name=f'CA0_wyh_{hidden_size}',omit_char=omit_char,device=device)
        self.ca_beta=nn.Sequential(
            nn.Linear(94,hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.ca_factor=nn.Sequential(
            nn.Linear(94,hidden_size)
        )
        
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.criterion=nn.MSELoss().to(device)
        
class CA1_wyh(CA_base_wyh):
    def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,omit_char=[],device='cuda'):
        CA_base_wyh.__init__(self,name=f'CA1_wyh_{hidden_size}',omit_char=omit_char,device=device)
        self.ca_beta=nn.Sequential(
            nn.Linear(94,32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32,hidden_size),
        )
        self.ca_factor=nn.Sequential(
            nn.Linear(94,hidden_size)
        )
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.criterion=nn.MSELoss().to(device)
        
class CA2_wyh(CA_base_wyh):
    def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,omit_char=[],device='cuda'):
        CA_base_wyh.__init__(self,name=f'CA2_wyh_{hidden_size}',omit_char=omit_char,device=device)
        self.ca_beta=nn.Sequential(
            nn.Linear(94,32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16,hidden_size),
        )
        self.ca_factor=nn.Sequential(
            nn.Linear(94,hidden_size)
        )
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.criterion=nn.MSELoss().to(device)
        
class CA3_wyh(CA_base_wyh):
    def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,omit_char=[],device='cuda'):
        CA_base_wyh.__init__(self,name=f'CA3_wyh_{hidden_size}',omit_char=omit_char,device=device)
        self.ca_beta=nn.Sequential(
            nn.Linear(94,32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(8,hidden_size),
        )
        self.ca_factor=nn.Sequential(
            nn.Linear(94,hidden_size)
        )
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.criterion=nn.MSELoss().to(device)
        
        
# class VAE_base(CA_base_wyh):
#     def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,omit_char=[],device='cuda',hidden_dims=[32,16,8]):
#         nn.Module.__init__(self)
#         CA_base_wyh.__init__(self,name=f'VAE_{hidden_size}',omit_char=omit_char,device=device)
#         # self.vae=VAE_wyh(hidden_size,dropout_rate,lr,device,hidden_dims)
#         self.beta_decoder=None
#         self.beta_encoder=None
#         self.mu=None
#         self.logvar=None
#         self.ca_factor=None
        
#     def forward(self,beta_input,factor_input):
#         beta_reparam,beta_output,factor_output,mu,logvar=self.vae(beta_input,factor_input)
#         output=torch.mm(beta_output,factor_output.T)
#         return output,mu,logvar
    

    
#     def train_one_epoch(self):
#         self.train_loader=self._get_dataset(period='train')
#         epoch_loss=0
#         for i,(beta_nn_input,factor_nn_input,label) in enumerate(self.train_loader):
#             self.optimizer.zero_grad()
#             beta_nn_input=beta_nn_input.squeeze(0).T
#             factor_nn_input=factor_nn_input.squeeze(0).T
#             label=label.squeeze(0)
#             output,mu,logvar= self.forward(beta_nn_input, factor_nn_input)
#             print(self.criterion)
#             loss = self.criterion(factor_nn_input, output,mu,logvar)
            
#             loss.backward()
#             self.optimizer.step()
#             epoch_loss += loss.item()

#             if i % 100 == 0:
#                 # print(f'Batches: {i}, loss: {loss.item()}')
#                 pass

#         return epoch_loss / len(self.train_loader)
    
#     def _valid_one_epoch(self):
#         epoch_loss = 0.0
#         for i, (beta_nn_input, factor_nn_input, labels) in enumerate(self.valid_loader):
#             # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
#             # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
#             # labels reshape: (1, 94) -> (94, ) (1*N => N,)
#             beta_nn_input = beta_nn_input.squeeze(0).T
#             factor_nn_input = factor_nn_input.squeeze(0).T
#             labels = labels.squeeze(0)

#             output,mu,logvar = self.forward(beta_nn_input, factor_nn_input)
#             loss = self.criterion(factor_nn_input, output,mu,logvar)
#             epoch_loss += loss.item()

#         return epoch_loss / len(self.valid_loader)

class VAE_wyh(CA_base_wyh):
    def __init__(self,hidden_size,dropout_rate=0.1,lr=0.01,device='cuda',omit_char=[],hidden_dims=[64,32,16]):
        CA_base_wyh.__init__(self,name=f'VAE2_wyh_{hidden_size}_lr_scheduler',omit_char=omit_char,device=device)
        self.beta_encoder=nn.Sequential(
            nn.Linear(94,hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0],hidden_size)
        )
        self.mu=nn.Linear(hidden_size,hidden_size)
        self.logvar=nn.Linear(hidden_size,hidden_size)
        # self.beta_decoder=nn.Sequential(
        #     nn.Linear(hidden_dims[2],hidden_size),
        # )
        
        self.ca_factor=nn.Sequential(
            nn.Linear(94,hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1],hidden_size),
        )
        self.lr=lr
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.lr_scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=15,gamma=0.1)
        self.criterion=self.criterion_
        self.reparameter=self.reparameter_
    def reparameter_(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std
    
    def forward(self,beta_input,factor_input):
        beta_latent=self.beta_encoder(beta_input)
        mu=self.mu(beta_latent)
        logvar=self.logvar(beta_latent)
        beta_reparam=self.reparameter(mu,logvar)
        plt.figure(figsize=(10,10))
        plt.hist(beta_reparam.flatten().detach().cpu().numpy(), bins=50)
        plt.savefig('./imgs/beta_latent.png')
        # beta_output=self.beta_decoder(beta_reparam)
        beta_output=beta_reparam
        factor_output=self.ca_factor(factor_input)
        # print(beta_output.shape,factor_output.shape)
        output=torch.sum(beta_output*factor_output,dim=1)
        return output,mu,logvar,beta_output
        # return beta_reparam,beta_output,factor_output,mu,logvar
    
    def criterion_(self,factor_input,output,mu,logvar):
        # print(factor_input,output)
        # beta= 3.0
        recon_loss=F.mse_loss(factor_input,output,reduction='mean')
        kl_loss= -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
        # print('kl_loss',kl_loss)
        # breakpoint()
        
        return recon_loss+ 100*kl_loss,recon_loss,kl_loss
    
    def train_one_epoch(self):
        self.train_loader=self._get_dataset(period='train')
        epoch_loss=0
        recon_loss,kl_loss=0,0
        for i,(beta_nn_input,factor_nn_input,label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            beta_nn_input=beta_nn_input.squeeze(0).T
            factor_nn_input=factor_nn_input.squeeze(0).T
            label=label.squeeze(0)
            # print(beta_nn_input.shape,factor_nn_input.shape,label.shape)
            output,mu,logvar,beta_output= self.forward(beta_nn_input, factor_nn_input)
            # print(self.criterion)
            loss,recon,kl = self.criterion(factor_nn_input, output,mu,logvar)
            recon_loss+=recon.item()
            kl_loss+=kl.item()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if i % 100 == 0:
                # print(f'Batches: {i}, loss: {loss.item()}')
                pass
        self.lr_scheduler.step()
        print('recon',recon_loss/len(self.train_loader),'kl',kl_loss/len(self.train_loader))
        # print('lr ',self.optimizer.param_groups[0]['lr'])
        return epoch_loss / len(self.train_loader)
    
    def _valid_one_epoch(self):
        epoch_loss = 0.0
        for i, (beta_nn_input, factor_nn_input, labels) in enumerate(self.valid_loader):
            # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
            # labels reshape: (1, 94) -> (94, ) (1*N => N,)
            beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)

            output,mu,logvar,beta_output = self.forward(beta_nn_input, factor_nn_input)
            loss,_ ,_= self.criterion(factor_nn_input, output,mu,logvar)
            epoch_loss += loss.item()

        return epoch_loss / len(self.valid_loader)
    
    def reset_parameters(self):
        super().reset_parameters()
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] = self.lr
        
if __name__=='__main__':
    model=CA_base_wyh()
    model.train_one_epoch()
    