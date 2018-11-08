import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from IPython.display import clear_output
from time import time
import gc

def standard_scale(col):

    try:
        return col.apply(lambda x:(x-col.mean())/col.std())
    except:
        return col.apply(lambda x:np.nan)
    
def process_log(num):
    
    return np.sign(num)*np.log(abs(num)+1)

def get_weighted(df,id_name,count_col,cols):
    
    counts = df.groupby(id_name).count()[count_col]
    df_weights = df.copy()
    df_weights['weights'] = np.concatenate([process_log(np.array(range(1,i+1))) for i in counts])
    weight_sums = df_weights.groupby(id_name).sum()['weights']
    df_weights['weight_sums'] = [a for a,b in zip(weight_sums,counts) for i in range(b)]
    for i in df_weights[list(cols)].columns:
        df_weights[i] = df_weights[i]*df_weights['weights']/df_weights['weight_sums']
    df_weights = df_weights.groupby(id_name).sum()[cols]
    df_weights.columns = [str(i)+'_weighted' for i in df_weights.columns]
    return df_weights

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from IPython.display import clear_output
from time import time
import gc

def get_coefs(df,x_col,y_col):
    
    try:
        model = LinearRegression().fit(df[x_col].values.reshape(-1,1),df[y_col].values.reshape(-1,1))
        return model.coef_[0][0]
    except:
        return np.nan

def get_slopes(df, skids, id_name, x_col, y_cols, progress = True):
    
    slopes, no_slopes, end_step = [],[np.nan for i in y_cols], len(skids)
    cols, t0 = list(y_cols)+list([x_col]), time()

    for step,i in enumerate(skids):
        dfs = df[df[id_name]==i][cols]
        if dfs.shape[0]>1:
            slopes.append([get_coefs(dfs[[x_col,j]].dropna(),x_col,j) for j in y_cols])
        else:
            slopes.append(no_slopes)
        if (step+1)%1000 == 0 or step == len(skids):
            dt = (time()-t0)/60
            clear_output(wait=True)
            print(round(step/end_step*100,3),'%')
            print(int((dt*(end_step/step - 1))))
    print('Found slopes in', y_cols, 'grouped by',id_name,'with respect to',x_col)

    slope_df = pd.DataFrame(slopes, index = skids, columns = [i+'_slope' for i in y_cols])
    slope_df.index.name = id_name
    del df,dfs
    gc.collect()
    return slope_df

def get_weighted_means(df, skids, id_name, cols, ascending = True, progress = True):
    
    weighted_means, step, end_step = [], 1, len(skids)
    t0 = time()
    for i in skids:
        df2 = df[df[id_name]==i][list(cols)]
        if ascending:
            df2['weights'] = list(map(process_log,list(range(1,df2.shape[0]+1))))
        else:
            df2['weights'] = list(map(process_log,list(range(1,df2.shape[0]+1))))[::-1]
        
        df2['weights'] = df2['weights']/df2.shape[0]
        df2_sum = sum(df2['weights'])
        df2['weights'] = df2['weights']/df2_sum
        df2[df2.columns[:-1]] = df2[df2.columns[:-1]].apply(lambda x:x*df2['weights'])
        
        df3 = pd.DataFrame([df2.drop('weights',axis=1).sum()])
        df3.index =[i]
        weighted_means.append(df3)
        if progress:
            step += 1
            if step%1000 == 0 or step == len(skids):
                dt = (time()-t0)/60
                clear_output(wait=True)
                print(round(step/end_step*100,3),'%')
                print(int((dt*end_step/step - dt)))

            
    weighted_means_df = pd.concat(weighted_means)
    weighted_means_df.columns = [[str(j)+'_weighted' for j in cols]]
    weighted_means_df.index.name = id_name
    del df,df2,df3
    gc.collect()
    return weighted_means_df
                
def drop_high_corrs(df, threshold):
    
    corr_df = df.corr().abs()                     
    upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(np.bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    print('Removing %d columns' % (len(drop_cols)),drop_cols)
                                 
    return df.drop(drop_cols,axis=1) 
                                 
def feats_with_labels(features):
    
    train = pd.read_csv('application_train.csv').set_index('SK_ID_CURR')
    labels = train['TARGET']
                                 
    X_with_labels = features.loc[labels[features.index].notnull()]
    labels = labels.loc[X_with_labels.index]
                                 
    return X_with_labels, labels
                              
def join_tuples_cols(df, join_string = None):
    
    columns = df.columns
    if join_string == None:
        return ['_'.join(i) for i in columns]
    else:
        return[str('_'+join_string).join(i) for i in columns]
    
def get_not_imp(features, model, threshold = 1):
    
    importances = pd.DataFrame(model.feature_importances_, index=bureau.columns, columns=['importance'])
    not_imp = importances[importances['importance'] < threshold].index
    print('%d Features with importances < %d' %(len(not_imp), threshold),list(not_imp))
    
    return list(not_imp)

def get_aggs_df(df, skid, agg_dict):
    
    dfs, step, end_step = [], 1, len(agg_dict)
    for col, agg_funcs in agg_dict.items():
        print('Aggregating',col,'with',agg_funcs)
        df_agg = df.groupby(skid).agg(agg_funcs)[[col]]
        df_agg.columns = join_tuples_cols(df_agg)
        dfs.append(df_agg)
        clear_output(wait=True)
        print(round(step/end_step*100,3),'%')
        step += 1
    del df
    gc.collect()
    return pd.concat(dfs,axis=1)
        
def reduce_cat(df, col, num_keep_cats, replacement):
    nan_cats = [cat for cat, count in df[col].value_counts()[num_keep_cats:].items()]
    return df[col].replace(nan_cats,replacement)
def cnt_percents(df,num,den):
    num_col, den_col = list(df[num]), list(df[den])
    l=[]
    for a,b in zip(num_col,den_col):
        try:
            l.append(a/b)
        except:
            l.append(np.nan)
    return l
# def most_recent(df, skids, id_name, x_col):
#     most_recents = []
#     for i in skids[:5]:
#         df2 = df[df[id_name]==i][list(x_col)]
#         num_rows = df2.shape[0]
#         most_recents.append(df2.iloc[num_rows-1])
#     display(pd.DataFrame(most_recents))
                      
                                 