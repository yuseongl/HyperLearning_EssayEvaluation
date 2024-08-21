
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def metrics(y_true, y_pred):
    scores={
    'MSE':[],
    'MAE':[],
    'R2SCORE':[]
    }
    
    scores['MSE'].append(mean_squared_error(y_true, y_pred))
    scores['MAE'].append(mean_absolute_error(y_true, y_pred))
    scores['R2SCORE'].append(r2_score(y_true, y_pred))
    
    return scores