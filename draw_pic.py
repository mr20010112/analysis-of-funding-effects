import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from journal_impact_calculation import get_all_metrics



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
    return Sum_IF,paper_num, co_authors

def drawing_all_data(my_data, year, author_id):
    data = my_data
    x = year
    plt.axvline(x=2020, color='k', linestyle='--') 
    for i in range(len(data)):
        y = data[i]
        if i <len(data)-1:
            plt.plot(x, y, label=author_id[i], alpha = 0.5)
        elif i == len(data)-1:
            plt.plot(x, y, color='r', label = author_id[i])

    
    plt.legend()
    plt.title('results')
    # 显示图形
    plt.show()

def compute_results(scopus_id, path):
    data = pd.read_csv(path)
    author = data['author_id'].to_list()
    author = author[:20]
    author.append(scopus_id)
    Sum_IF, paper_num, co_authors = get_index(author)
    return author, Sum_IF, paper_num, co_authors



if __name__ == "__main__":

    year = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    scopus_id = '56128400700'
    author_id, Sum_IF, paper_num, co_authors = compute_results(scopus_id, './co_authors_contrast.csv')
    print(author_id)
    drawing_all_data(co_authors,year, author_id)


    