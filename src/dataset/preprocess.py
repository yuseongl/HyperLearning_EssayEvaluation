from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .get_data import get_data_in_file
from .dim_reduction import get_svd_data
from ..embeding.text_embeding import Embedding

def preprocess(path, embedding=False):
    '''
    function for preprocessing raw-data.
    
    include:
    one-hot encoding
    sentence classiication
    enbedding processing
    
    input : path -> str
    output : input data(x) and target(y) for machine learning -> Dataframe
    '''
    
    df = get_data_in_file(path)
    
    df_data = df.set_index('id')
    data_idx = df_data.index
    
    df_data['essay']= df_data['essay'].str.replace('#@문장구분#', '')
    
    le = LabelEncoder()
    df_data['grade'] = le.fit_transform(df_data['grade']) #라벨 인코더
    df_data = pd.get_dummies(df_data, columns=['etype']) #원-핫 인코딩

    col = {'etype_글짓기': '글짓기', 
        'etype_대안제시': '대안제시',
        'etype_설명글': '설명글',
        'etype_주장': '주장',
        'etype_찬성반대': '찬성반대'}
    df_data.rename(columns=col, inplace=True)
    
    if embedding == True:
        Embedding_ = Embedding()
        df_emb_data = Embedding_.make_emb(df_data, path)
    else:
        df_emb_data = pd.read_csv(path+'embedding_data'+'.csv')
    
    df_data.reset_index(inplace=True)
    main_df = get_svd_data(df_data, df_emb_data)
    main_df.set_index('id', inplace=True)
    
    df_x = main_df.loc[data_idx]
    y = round(df_x['scores'], 2)
    X = df_x.drop(columns=['scores'])

    X['grade']=X['grade'].astype('int')
    X['levels']=X['levels'].astype('int')
    
    return X, y

    