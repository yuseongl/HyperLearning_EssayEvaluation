import pandas as pd
import json

from os import listdir
from tqdm import tqdm

def get_data(data, path:str):
    '''
    function for getting attribute in raw-data.
    if you input location information can take attribute include 
    : 'id','grade','etype','essay','levels','scores'.
    
    input : rawdata location -> DataFrame
    output : attribute -> DataFrame
    '''
    
    ids, grades, etypes, essay, levels, scores =[], [], [], [], [], []  # 빈 리스트 생성 
    
    for row in tqdm(data.itertuples(), total=data.shape[0]):  # 진행상황 확인을 위한 tqdm, data frame 값을 빠르게 가져오기 위한 itertuples 
        file_str = path + row.file_name                       # 파일명 
        
        with open(file_str) as f:      # json파일 열기
            text = json.load(f)
        
        paragraph_score = text['score']['paragraph_score'][0]['paragraph_scoreT_avg'] # paragraph score data 가져오기 
        essay_score = text['score']['essay_scoreT_avg']                               # essay score data 가져오기 
        score = paragraph_score + essay_score    # score 더하기 

        ids.append(text['info']['essay_id'])         # essay_id
        grades.append(text['rubric']['essay_grade']) # 학년 
        etypes.append(text['info']['essay_type'])    # 글 종류
        essay.append(text['paragraph'][0]['paragraph_txt'])
        levels.append(text['info']['essay_level'])   # 각 글의 level
        scores.append(score)                         # score data 

    df= pd.DataFrame({'id': ids, 'grade': grades, 'etype':etypes, 'essay':essay, 'levels': levels, 'scores': scores})  # 리스트로 dataframe 만들기 
    return df 

def get_data_in_file(path ='data/1.Training/라벨링데이터/'):
    '''
    function for taking attribute in raw-data of every file
    
    input : path information -> str
        default location is 'data/1.Training/라벨링데이터/'
    output : attribute -> DataFrame
    '''
    
    text_type = ['글짓기','대안제시', '설명글', '주장', '찬성반대']   # 글 5가지 종류 
    main_df = pd.DataFrame()    # 빈 데이터프레임 

    # 파일 리스트로부터 모든 데이터 가져오기 
    for i in range(5):
        paths = path + text_type[i]+'/'   # 경로 지정 
        fileNameList = listdir(paths)      # 해당 경로에 포함된 파일 리스트를 모두 가져옴

        df= pd.DataFrame(fileNameList, columns=['file_name'])     # 파일 리스트 -> 데이터 프레임화 
        df=df.astype('string')                                    # df type string 으로 변경 
        
        cr = df['file_name'].str.contains('중등')                 # 파일명에 '중등'이 포함되어있는 파일만 filtering
        data = df[cr]

        sub_df = get_data(data, paths)                          # 데이터 가져오는 함수
        main_df = pd.concat([main_df, sub_df])            # main_labels에 누적하여 더하기 
        
    return main_df