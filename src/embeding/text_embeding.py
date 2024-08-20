import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertTokenizer,TFBertModel, BertModel, AutoModel
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backend.mps.is_available() else 'cpu'
#device = 'cpu'

class Embedding:
    '''
    function for making embedding feature using bert model.
    this use pretrained klue bert model and tokenizer
    
    input : original data -> dataframe 
    output : data added embeding feature -> Dataframe
    '''
    def __init__(self):
        # 토크나이저 
        self.tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
        # 모델
        self.model = TFBertModel.from_pretrained("klue/bert-base", from_pt=True)
        
    def gen_emb(self, data):
        '''
        function for generating embeding vecter
        '''
        for row in tqdm(data.itertuples()):
            inputs = self.tokenizer(row.essay, return_tensors='tf', truncation=True)
            outputs = self.model(inputs)
            cls_embeddings = outputs.last_hidden_state[:,0]
            yield cls_embeddings 
    
    def make_emb(self, data, path):
        '''
        function for enbeding vector as dataframe 
        '''
        cls_emb = list(self.gen_emb(data))
        cls_emb_df = pd.DataFrame(np.squeeze(np.array(cls_emb), axis=1))
        cls_emb_df.to_csv(path+'embedding_data'+'.csv',index=False)
        
        return cls_emb_df