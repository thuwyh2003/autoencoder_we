import torch
from models.PCA import PCA
from models.FF import FF
from models.IPCA import IPCA
from models.CA import CA0, CA1, CA2, CA3
from models.CA_wyh import CA_base_wyh,CA0_wyh,CA1_wyh,CA2_wyh,CA3_wyh,VAE_wyh
import gc
import argparse
import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm
from utils import *
from analysis import *
import matplotlib.pyplot as plt
from itertools import product
import os

import warnings
warnings.filterwarnings('ignore')

def predict_and_inference(model):
    model = model.to('cuda')
    month_list=pd.read_pickle('data/mon_list.pkl')
    test_month=month_list.loc[(month_list>model.test_period[0])]
    
    test_bar=tqdm(test_month.groupby(test_month.apply(lambda x:x//10000)),colour='red',desc=f'Testing for {model.name}')
    
    if not len(model.omit_char):
        inference_result=pd.DataFrame()
        predict_result=pd.DataFrame()
    else: 
        inference_result=[]
    # print(test_month)
    # i=0
    for g in test_bar:
        # i+=1
        # if i>2:
        #     break
        test_bar.set_postfix({'Year':g[0]})
        
        model.reset_parameters()
        model.release_gpu()
        train_loss,val_loss=model.train_model()
        plt.figure(figsize=(10,5))
        plt.plot(train_loss[5:],label='train_loss')
        plt.plot(val_loss[5:],label='val_loss')
        plt.legend()
        plt.savefig(f'results/{model.name}_loss_{g[0]}.png')
        plt.close()
        for m in g[1].to_list():
            # print(g[1].to_list())
            m_stock_index,_,_,_=model._get_item(m)
            m_stock_index=pd.Series(m_stock_index)
            
            if not len(model.omit_char):
                inference_R=model.inference(m)
                inference_R=inference_R.cpu().detach().numpy()
                inference_R=pd.DataFrame(inference_R,index=m_stock_index,columns=[m])
                inference_result=pd.concat([inference_result,inference_R],axis=1)
                
                predict_R=model.predict(m)
                predict_R=predict_R.cpu().detach().numpy()
                predict_R=pd.DataFrame(predict_R,index=m_stock_index,columns=[m])
                predict_result=pd.concat([predict_result,predict_R],axis=1)
            else:
                inference_R=model.inference(m)
                inference_result.append(inference_R)    #   T,N,m
            gc.collect()
        model.refit()
    if not len(model.omit_char):    
        inference_result=pd.DataFrame(inference_result.values.T,index=test_month,columns=CHARAS_LIST)
        predict_result=pd.DataFrame(predict_result.values.T,index=test_month,columns=CHARAS_LIST)
        inference_result.to_csv(f'results/inference/{model.name}_inference.csv')
        predict_result.to_csv(f'results/predict/{model.name}_predict.csv')
        
    del model
    gc.collect()
    return inference_result       
    
def call_model(model_name,omit_char,k_list):
    assert model_name in ['CA0','CA1','CA2','CA3','VAE']
    if model_name=='CA0':
        return[{
                'name':f'CA0_{k}',
                'omit_char': omit_char,
                'model':CA0_wyh(hidden_size=k,lr=0.001,omit_char=omit_char,device='cuda')
            }   for k in k_list]
    if model_name=='CA1':
        return[{
                'name':f'CA1_{k}',
                'omit_char': omit_char,
                'model':CA1_wyh(hidden_size=k,lr=0.001,omit_char=omit_char,device='cuda')   
            }   for k in k_list]
    if model_name=='CA2':
        return[{
                'name':f'CA2_{k}',
                'omit_char': omit_char,
                'model':CA2_wyh(hidden_size=k,lr=0.001,omit_char=omit_char,device='cuda')
            }   for k in k_list]
    if model_name=='CA3':
        return[{
                'name':f'CA3_{k}',
                'omit_char': omit_char,
                'model':CA3_wyh(hidden_size=k,lr=0.001,omit_char=omit_char,device='cuda')
            }   for k in k_list]
    if model_name=='VAE':
        return[{
            'name':f'VAE_{k}',
            'omit_char': omit_char,
            'model':VAE_wyh(hidden_size=k,dropout_rate=0.1,lr=0.01,omit_char=omit_char,device='cuda')
        }  for k in k_list
        ]        
         
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='VAE')
    parser.add_argument('--k_list',type=int,nargs='+',default=[5,4,3,1])
    parser.add_argument('--omit_char', type=str, default='')
    args=parser.parse_args()
    print(args.k_list)
    if 'results' not in os.listdir('./'):
        os.mkdir('results')
    if 'inference' not in os.listdir('./results'):
        os.mkdir('results/inference')
    if 'predict' not in os.listdir('./results'):
        os.mkdir('results/predict')
    if 'imgs' not in os.listdir('./'):
        os.mkdir('imgs')
    R_square=[]
    models_name=[]
    model_list=args.model.split(',')
    for model_name in model_list:
        if isinstance(args.omit_char, str) and len(args.omit_char) > 0:
            omit_chars = args.omit_char.split(' ')
        else:
            omit_chars = []
        model_ks=call_model(model_name,omit_chars,k_list=args.k_list)
        
        
        for model in model_ks:
            models_name.append(model['name'])
            print(f"{time.strftime('%a, %d %b %Y %H:%M:%S +0800', time.gmtime())} | Model: {model['name']} | {omit_chars}")
            # print(model['model'])
            model['model'].to("cuda")
            inference_result=predict_and_inference(model['model'])
            
            if not len(model['omit_char']):
                R_square.append(calculate_R2(model['model'], 'inference'))
                alpha_plot(model['model'], 'inference', save_dir='imgs')
            else:
                print(inference_result)
                inf_ret = np.array(inference_result)
                print(inf_ret.shape)
                print(model['omit_char'])
                for i in range(len(model['omit_char'])):
                    inference_r = inf_ret[:, :, i] # T * N
                    complete_r = inf_ret[:, :, -1]
                    R_square.append(calculate_R2(None, None, inference_r, complete_r))
            
    p = time.localtime()
    time_str = "{:0>4d}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}".format(p.tm_year, p.tm_mon, p.tm_mday, p.tm_hour, p.tm_min, p.tm_sec)
    filename = f"R_squares/{time_str}.json"
    obj = {
        "models": models_name,
        'omit_char': args.omit_char.split(' '),
        "R2_total": R_square,
    }

    with open(filename, "w") as out_file:
        json.dump(obj, out_file)