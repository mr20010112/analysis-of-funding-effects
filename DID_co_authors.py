import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import statsmodels.formula.api as smf


#from data_process import compute_author_paper
from collections import defaultdict
from journal_impact_calculation import get_all_metrics
# from pybliometrics.scopus.utils import config

import matplotlib.pyplot as plt

def get_top_L_similar_scholars_from_Shenzhen_data(scopus_id, L):
    res_list = []
    CN_matrix = np.loadtxt('./Data/Shenzhen_Scholar_Network/CN_matrix_of_scholars.txt')
    scholar_nodes = np.load('./Data/Shenzhen_Scholar_Network/scholar_list.npy').tolist()
    line = CN_matrix[:, scholar_nodes.index(int(scopus_id))]
    indices = np.argsort(line)[-L: ]
    for each_idx in indices:
        res_list.append(scholar_nodes[each_idx])
    print(res_list)
    return res_list




def get_index(author_id):

    paper_num = []
    Sum_JCR_Q1 = []
    Sum_ZKY = []
    Avg_IF = []
    Sum_IF = []
    Max_IF = []
    Sum_JCR = []
    Avg_JCR = []
    Max_JCR = []
    co_authors = []
    
    for ele in author_id:
        ele = str(ele)
        list_of_years, res_dict = get_all_metrics(ele)
        paper_num.append(res_dict['年累计发表论文数'])
        Sum_JCR_Q1.append(res_dict['年度累计发表JCR分区Q1论文数量'])
        Sum_ZKY.append(res_dict['年度累计发表中科院分区顶级期刊论文数量'])
        Avg_IF.append(res_dict['年度平均IF'])
        Sum_IF.append(res_dict['年度累计IF'])
        Max_IF.append(res_dict['年度最高IF'])
        Sum_JCR.append(res_dict['年度累计JCI'])
        Avg_JCR.append(res_dict['年度平均JCI'])
        Max_JCR.append(res_dict['年度最高JCI'])
        co_authors.append(res_dict['年度累计合作者数量'])
    

    return paper_num, Sum_JCR_Q1, Sum_ZKY, Avg_IF, Sum_IF, Max_IF, Sum_JCR, Avg_JCR, Max_JCR, co_authors

'''
prop值为aper_num, Sum_JCR_Q1, Sum_ZKY, Avg_IF, Sum_IF, Max_IF, Sum_JCR, Avg_JCR, Max_JCR
df为传入的空dataframe
year为传入的年份
'''
def create_df_by_index_all(df, p1, year, author_id):

    iteration = len(p1)
    print(iteration)
    df_contrast = pd.DataFrame()
    id = []
    y = []
    p = []
    for i in range(iteration):
        v = author_id[i]
        id = id + [v for j in range(len(year))]
        p = p + p1[i]
  
    y = year*len(author_id)
    data = {

        'id': id,
        'year': y,
        'co_authors': p
    }

    df = pd.DataFrame(data) 
    df['after'] = np.where(df['year'] >= '2020', 1, 0)

    return df

def create_df_by_index(df, year, author_id):

    v = author_id
    id = [v for j in range(len(year))]
    author_id_list = []
    author_id_list.append(author_id)
    paper_num, Sum_JCR_Q1, Sum_ZKY, Avg_IF, Sum_IF, Max_IF, Sum_JCR, Avg_JCR, Max_JCR, co_authors = get_index(author_id_list)

    data = {

        'id': id,
        'year': year,
        'paper_num': paper_num[0],
        'Sum_JCR_Q1': Sum_JCR_Q1[0],
        'Sum_ZKY': Sum_ZKY[0],
        'Avg_IF': Avg_IF[0],
        'Sum_IF': Sum_IF[0],
        'Max_IF': Max_IF[0],
        'Sum_JCR': Sum_JCR[0],
        'Max_IF': Max_IF[0],
        'Sum_JCR': Sum_JCR[0],
        'Avg_JCR': Avg_JCR[0],
        'Max_JCR': Max_JCR[0],
        'co_authors':co_authors[0]
    }

    df = pd.DataFrame(data) 
    df['after'] = np.where(df['year'] >= '2020', 1, 0)

    return df



def fit_model_by_paper_num_and_get_summary(new_df):
    before_treat = new_df[new_df['after'] == 0]
    model = smf.ols('co_authors ~ treatment*year', data=before_treat).fit()
    return model


def drawing_all_data(my_data, year, author_id):
    data = my_data
    x = year 
    for i in range(len(data)):
        y = data[i]
        if i ==0 :
            plt.plot(x, y, color='r', label = author_id[i])
        else:
            plt.plot(x, y, label=author_id[i],linestyle = ':')

    plt.axvline(x='2020', color='k', linestyle='--')
    plt.legend()
    plt.title('results')
    # 显示图形
    plt.show()

def DID(new_df, path):
    new_df = pd.read_csv(path, dtype={'id': np.int32, 'year': str, 'treatment': np.int32, 'after': np.int32})

    model = smf.ols('co_authors ~ treatment*year', data=new_df).fit()
    return model

def DID_after(new_df, path):
    new_df = pd.read_csv(path, dtype={'id': np.int32, 'year': str, 'treatment': np.int32, 'after': np.int32})
    model = smf.ols('co_authors ~ treatment*after', data=new_df).fit()
    return model


def train_and_contrast(train_id, contrast_id):
    year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    pre_year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
    paper_num, Sum_JCR_Q1, Sum_ZKY, Avg_IF, Sum_IF, Max_IF, Sum_JCR, Avg_JCR, Max_JCR, co_authors = get_index(contrast_id)

     #构造实验组
    df_train = pd.DataFrame()
    df_train = create_df_by_index(df_train, year, train_id)
    df_train['treatment'] = 1

    #构造对照组
    df_contrast = pd.DataFrame()
    df_contrast = create_df_by_index_all(df_contrast, co_authors, year, contrast_id)
    df_contrast['treatment'] = 0
    
    df = pd.concat([df_train, df_contrast], axis=0, ignore_index=True)
    selected_columns = ['id', 'year', 'co_authors', 'after', 'treatment']
    new_df = df.loc[:, selected_columns]
    return new_df

def draw_parallel(summary):
    coefficients = summary.params
    coef = [coefficients['treatment:year[T.2013]'], coefficients['treatment:year[T.2014]'], coefficients['treatment:year[T.2015]'], coefficients['treatment:year[T.2016]'],coefficients['treatment:year[T.2017]'],coefficients['treatment:year[T.2018]'],coefficients['treatment:year[T.2019]'],coefficients['treatment:year[T.2020]'],coefficients['treatment:year[T.2021]'],coefficients['treatment:year[T.2022]']]

    bse = summary.bse
    se = [bse['treatment:year[T.2013]'], bse['treatment:year[T.2014]'], bse['treatment:year[T.2015]'], bse['treatment:year[T.2016]'],bse['treatment:year[T.2017]'],bse['treatment:year[T.2018]'],bse['treatment:year[T.2019]'], bse['treatment:year[T.2020]'],bse['treatment:year[T.2021]'],bse['treatment:year[T.2022]']]

    data = {
    'year': ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'],
    'interaction_coef': coef,  # 交互项系数
    'std_error':se  # 标准误
    }
    df = pd.DataFrame(data)

    # 绘制误差棒图
    plt.errorbar(df['year'], df['interaction_coef'], df['std_error'], fmt='o', capsize=5)
    plt.plot(df['year'], df['interaction_coef'], marker='o', linestyle='-', color='blue')
    plt.axvline(x='2020', color='k', linestyle='--')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('year')
    plt.ylabel('interaction coefficient')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']

    train_id = '24824532900'
    contrast_id = ['56421165300', '56423261700', '56393916300', '56411251400', '56580731300', '56493248300', '56488693800', '56525773200', '56381779900','56163856700', '56176657400', '56161393800', '56102445200', '56118040000', '57075349600', '56982091400', '56942883900', '57187639800']
    new_df = train_and_contrast(train_id, contrast_id)
    new_df.to_csv('./co_authors_contrast_final.csv')
    
    y = new_df['co_authors'].to_list()
    n = [train_id] + contrast_id
    y_2d = [y[i*11:(i+1)*11] for i in range(len(n))]

    mode = int(input("请输入1，2，3："))
    if mode == 1:
        drawing_all_data(y_2d, year, n)
    elif mode == 2:
        results = DID(new_df,'./co_authors_contrast_final.csv')
        draw_parallel(results)
    
    else:
        results = DID_after(new_df,'./co_authors_contrast_final.csv')
        print(results.summary())
    

        
        




        



    


    





   
    
