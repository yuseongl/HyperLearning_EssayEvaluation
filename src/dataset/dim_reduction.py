import pandas as pd
from sklearn.decomposition import TruncatedSVD

def svd(data):
    svd = TruncatedSVD(n_components=200)   # 200은 임의로 선정한 값
    emb_svd = svd.fit_transform(data)
    df_emb = pd.DataFrame(emb_svd)
    
    return df_emb

def svd_concat_data(data, emb_data, reduction=False):
    data.reset_index(inplace=True)
    if reduction == True:
        emb_data = svd(emb_data)
        
    df = pd.concat([data, emb_data], axis=1)
    df.drop(columns=['essay'], inplace=True) # text data는 임베딩 값으로 대체되므로 삭제 
    
    return df