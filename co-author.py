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
    Sum_IF = []
    co_authors = []
    
    for ele in author_id:
        ele = str(ele)
        list_of_years, res_dict = get_all_metrics(ele)
        paper_num.append(res_dict['年累计发表论文数'])
        Sum_IF.append(res_dict['年度累计IF'])
        co_authors.append(res_dict['年度累计合作者数量'])
    

    return paper_num, Sum_IF,co_authors



def create_df_by_index(df, year, author_id):

    v = author_id
    id = [v for j in range(len(year))]
    author_id_list = []
    author_id_list.append(author_id)
    paper_num, Sum_IF, co_authors = get_index(author_id_list)

    data = {

        'id': id,
        'year': year,
        'paper_num': paper_num[0],
        'Sum_IF': Sum_IF[0],
        'co_authors': co_authors[0]
    }

    df = pd.DataFrame(data) 
    df['after'] = np.where(df['year'] >= 2020, 1, 0)

    return df



def fit_model_by_Sum_IF_and_get_summary(new_df):
    before_treat = new_df[new_df['after'] == 0]
    model = smf.ols('co_authors ~ treatment*year', data=before_treat).fit()

    return model

def test_passing(scopus_id):
    #scopus_id = '24824532900'
    #最大为13150
    L = 13150
    author_id = get_top_L_similar_scholars_from_Shenzhen_data(scopus_id, L)
    #year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    #pre_year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

    year = [ i for i in range(2012,2023)]
    pre_year = [i for i in range(2012,2021)]

    #构造实验组

    df_train = pd.DataFrame()
    df_train = create_df_by_index(df_train, year, scopus_id)
    df_train['treatment'] = 1

    col = ['author_id', 'coefficients', 't_values', 'p_values', 'r_squared']
    suss_summaries = pd.DataFrame(columns=col)



    for i in range(len(author_id)):

        #构造对照组
        df_contrast = pd.DataFrame()
        df_contrast = create_df_by_index(df_contrast, year, author_id[i])
        df_contrast['treatment'] = 0



        df = pd.concat([df_train, df_contrast], axis=0, ignore_index=True)
        
        selected_columns = ['id', 'year', 'co_authors', 'after', 'treatment']
        new_df = df.loc[:, selected_columns]
        print(new_df)

        summary = fit_model_by_Sum_IF_and_get_summary(new_df)

        print (summary.summary())
        
        #提取常见的统计信息
        coefficients = summary.params
        print(coefficients)

        se = summary.bse

        t_values = summary.tvalues

        p_values = summary.pvalues
        pre_p_values = p_values['treatment:year']

        
        r_squared = summary.rsquared

        #f_statistic = summary.tables[0].data[3][3]

        

        
        # 检查所有p值是否接近1
        if pre_p_values>0.9:
            print(1)
            # 如果所有的p值都大于0.05，那么添加True到结果列表，因为这表明我们没有证据拒绝平行趋势假设
            new_row = [[author_id[i], coefficients, t_values, p_values, r_squared]]
            new_dataframe = pd.DataFrame(new_row, columns=suss_summaries.columns)
            suss_summaries = pd.concat([suss_summaries, new_dataframe], ignore_index=True)

    return suss_summaries
        
        

    #print(suss_summaries)
    #suss_summaries.to_csv('./Sum_IF_contrast.csv')

if __name__ == "__main__":
        
    
    #print(suss_summaries)
    suss_summaries = test_passing('56128400700')
    suss_summaries.to_csv('./co_authors_contrast1.csv')

