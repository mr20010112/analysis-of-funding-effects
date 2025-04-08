import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import pymannkendall as mk
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.formula.api as smf


#from data_process import compute_author_paper
from pathlib import Path
from collections import defaultdict
from journal_impact_calculation import get_all_metrics
# from pybliometrics.scopus.utils import config

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def get_index(author_id):

    paper_num = []
    Sum_IF = []
    Sum_JCR = []
    co_authors = []
    
    for ele in author_id:
        ele = str(ele)
        list_of_years, res_dict = get_all_metrics(ele)
        paper_num.append(res_dict['年累计发表论文数'])
        Sum_IF.append(res_dict['年度累计IF'])
        Sum_JCR.append(res_dict['年度累计JCI'])
        co_authors.append(res_dict['年度累计合作者数量'])
    

    return paper_num, Sum_IF, Sum_JCR, co_authors


def create_df_by_index(df, year, did_year, author_id):

    v = author_id
    id = [v for j in range(len(year))]
    author_id_list = []
    author_id_list.append(author_id)
    paper_num, Sum_IF, Sum_JCR, co_authors = get_index(author_id_list)

    data = {

        'id': id,
        'year': year,
        'paper_num': paper_num[0],
        'Sum_IF': Sum_IF[0],
        'Sum_JCR': Sum_JCR[0],
        'co_authors':co_authors[0]
    }

    df = pd.DataFrame(data) 
    df['after'] = np.where(df['year'] >= did_year, 1, 0)

    return df

def create_df_by_index_all(df, p1, p2, p3, p4, year, did_year, author_id):

    iteration = len(p1)
    print(iteration)
    df_contrast = pd.DataFrame()
    id = []
    y = []
    p1_all = []
    p2_all = []
    p3_all = []
    p4_all = []
    for i in range(iteration):
        v = author_id[i]
        id = id + [v for j in range(len(year))]
        p1_all = p1_all + p1[i]
        p2_all = p2_all + p2[i]
        p3_all = p3_all + p3[i]
        p4_all = p4_all + p4[i]

  
    y = year*len(author_id)
    data = {

        'id': id,
        'year': y,
        'paper_num': p1_all,
        'Sum_IF':p2_all,
        'Sum_JCR':p3_all,
        'co_authors': p4_all,
    }

    df = pd.DataFrame(data) 
    df['after'] = np.where(df['year'] >= did_year, 1, 0)

    return df

def drawing_all_data(my_data, year, did_year, author_id, save_path=None):
    # 设置支持中文的字体，使用系统中已有的 'Noto Sans CJK JP'
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 指定支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    data = my_data
    x = year 
    for i in range(len(data)):
        y = data[i]
        if i == 0:
            plt.plot(x, y, color='r', label=author_id[i])
        else:
            plt.plot(x, y, label=author_id[i], linestyle=':')

    plt.axvline(x=did_year, color='k', linestyle='--')
    plt.legend()
    plt.title('结果')  # 中文标题

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # 显示图形
    plt.show()

def DID(new_df, weights, type, path):
    new_df = pd.read_csv(path, dtype={'id': np.int32, 'year': str, 'treatment': np.int32, 'after': np.int32})
    
    # 创建与new_df行数匹配的权重数组
    full_weights = np.ones(len(new_df))
    # 实验组权重设为1
    # 对照组权重根据作者ID分配
    for i, author_id in enumerate(np.unique(new_df[new_df['treatment'] == 0]['id'])):
        if i < len(weights):
            full_weights[new_df['id'] == author_id] = weights[i]
    
    model = smf.wls(f'{type} ~ treatment*year', data=new_df, weights=full_weights).fit()
    return model

def DID_after(new_df, weights, type, data_path, save_path=None):
    # 读取数据
    new_df = pd.read_csv(data_path, dtype={'id': np.int32, 'year': str, 'treatment': np.int32, 'after': np.int32})
    
    # 创建与new_df行数匹配的权重数组
    full_weights = np.ones(len(new_df))
    # 实验组权重设为1
    # 对照组权重根据作者ID分配
    for i, author_id in enumerate(np.unique(new_df[new_df['treatment'] == 0]['id'])):
        if i < len(weights):
            full_weights[new_df['id'] == author_id] = weights[i]
    
    # 拟合加权最小二乘模型
    model = smf.wls(f'{type} ~ treatment*after', data=new_df, weights=full_weights).fit()
    
    # 提取模型结果的关键信息
    summary = model.summary()
    coef_table = summary.tables[1]  # 系数表格是第二个表格
    data = pd.DataFrame(coef_table.data[1:], columns=coef_table.data[0])  # 转换为 DataFrame
    
    # 将英文列名和变量名翻译为中文
    col_names = {
        '': '变量',
        'coef': '系数',
        'std err': '标准误',
        't': 't值',
        'P>|t|': 'p值',
        '[0.025': '置信区间下限',
        '0.975]': '置信区间上限'
    }
    var_names = {
        'Intercept': '截距',
        'treatment': '处理',
        'after': '处理后',
        'treatment:after': '处理×处理后'
    }
    
    # 替换列名和变量名
    data.columns = [col_names[col] for col in data.columns]
    data['变量'] = [var_names.get(var, var) for var in data['变量']]
    
    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 使用支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 4))  # 设置图片大小
    ax.axis('off')  # 隐藏坐标轴
    
    # 绘制表格
    table = ax.table(cellText=data.values, 
                     colLabels=data.columns, 
                     cellLoc='center', 
                     loc='center')
    
    # 调整表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)  # 调整表格大小
    plt.title('双重差分模型结果', fontsize=16, pad=20)
    
    # 保存图片（如果提供了 save_path）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'./{type}_did_result.png', dpi=300, bbox_inches='tight')  # 默认保存路径
    
    # 显示图片
    plt.show()
    
    # 仍然返回原始的 summary 以保持函数兼容性
    return model

def train_and_contrast(train_id, contrast_id, did_year, type):
    year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    pre_year = [y for y in year if int(y) < int(did_year)]
    paper_num, Sum_IF, Sum_JCR, co_authors = get_index(contrast_id)

     #构造实验组
    df_train = pd.DataFrame()
    df_train = create_df_by_index(df_train, year, did_year, train_id)
    df_train['treatment'] = 1

    #构造对照组
    df_contrast = pd.DataFrame()
    df_contrast = create_df_by_index_all(df_contrast, paper_num, Sum_IF, Sum_JCR, co_authors, year, did_year, contrast_id)
    df_contrast['treatment'] = 0
    
    df = pd.concat([df_train, df_contrast], axis=0, ignore_index=True)
    selected_columns = ['id', 'year', type, 'after', 'treatment']
    new_df = df.loc[:, selected_columns]
    return new_df

def draw_parallel(summary, did_year, save_path=None):
    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 使用支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    coefficients = summary.params
    year = ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    coef = [
        coefficients['treatment:year[T.2013]'], coefficients['treatment:year[T.2014]'],
        coefficients['treatment:year[T.2015]'], coefficients['treatment:year[T.2016]'],
        coefficients['treatment:year[T.2017]'], coefficients['treatment:year[T.2018]'],
        coefficients['treatment:year[T.2019]'], coefficients['treatment:year[T.2020]'],
        coefficients['treatment:year[T.2021]'], coefficients['treatment:year[T.2022]']
    ]

    bse = summary.bse
    se = [
        bse['treatment:year[T.2013]'], bse['treatment:year[T.2014]'],
        bse['treatment:year[T.2015]'], bse['treatment:year[T.2016]'],
        bse['treatment:year[T.2017]'], bse['treatment:year[T.2018]'],
        bse['treatment:year[T.2019]'], bse['treatment:year[T.2020]'],
        bse['treatment:year[T.2021]'], bse['treatment:year[T.2022]']
    ]

    data = {
        'year': ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'],
        'interaction_coef': coef,  # 交互项系数
        'std_error': se  # 标准误
    }
    df = pd.DataFrame(data)

    # 绘制误差棒图
    plt.errorbar(df['year'], df['interaction_coef'], df['std_error'], fmt='o', capsize=5)
    plt.plot(df['year'], df['interaction_coef'], marker='o', linestyle='-', color='blue')
    plt.axvline(x=did_year, color='k', linestyle='--')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('年份')  # 中文标签
    plt.ylabel('交互项系数')  # 中文标签
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def feature_extra(data):
    num_samples, time_steps, num_features = data.shape
    features = []

    alpha = 0.3

    for sample in range(num_samples):
        sample_features = []
        for feature in range(num_features):
            ts = data[sample, :, feature]

            mean = np.mean(ts)

            delta = np.max(ts) - np.min(ts)

            if time_steps < 2 or np.var(ts) == 0:
                slope = 0
            else:
                slope, intercept, _, _, _ = linregress(np.arange(time_steps), ts)

            ewma = pd.Series(ts).ewm(alpha=alpha).mean().iloc[-1]

            mk_result = mk.original_test(ts)
            tau = mk_result.Tau

            sample_features.append([mean, delta, slope, intercept, ewma, tau])

        features.append(sample_features)

    return np.array(features)

def compute_similar(contrast_data, ref_data):

    similarity = np.zeros((len(contrast_data)))
    for i in range(len(contrast_data)):
        for j in range(len(ref_data)):
            similarity[i] += np.mean(np.diag(cosine_similarity(contrast_data[i], ref_data[j])))

    return similarity/len(ref_data)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=11):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_dim, 1))  # (feature_dim, 1)
        self.softmax = nn.Softmax(dim=2)  # Softmax over feature dimension

    def forward(self, x):
        # x shape: (batch_size, time_length, feature_dim)
        attention_scores = torch.matmul(x, self.attention_weights)  # (batch_size, time_length, 1)
        attention_scores = self.softmax(attention_scores)  # (batch_size, time_length, 1)
        weighted_features = torch.sum(x * attention_scores, dim=2)  # (batch_size, time_length)
        return weighted_features

class PS_Estimation(nn.Module):
    def __init__(self, time_length=11, feature_dim=4, hidden_dim=32, device='cuda:0'):
        super(PS_Estimation, self).__init__()
        self.device = device
        self.feature_expand = nn.Linear(feature_dim, 2*feature_dim)
        self.positional_encoding = PositionalEncoding(2*feature_dim, max_len=time_length)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2*feature_dim, nhead=4),
            num_layers=2
        )
        self.attention_layer = AttentionLayer(2*feature_dim)
        self.fc1 = nn.Linear(time_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x = self.feature_expand(x)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # (time_length, batch_size, feature_dim) for transformer
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, time_length, feature_dim)
        x = self.attention_layer(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
    def fit_data(self, samples, labels, save_dir=None, load_dir=None, num_epochs=1, warm_up_epochs=0, batch_size=1, lr=1.0e-5):

        if load_dir is None:

            interval = math.ceil(samples.shape[0] / batch_size)
            total_steps = num_epochs * interval
            warm_up_steps = warm_up_epochs * interval
            main_steps = total_steps - warm_up_steps
            self.lr = lr
            self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)
            warm_up_scheduler = LinearLR(self.opt, start_factor=1e-8, end_factor=1.0, total_iters=warm_up_steps)
            cosine_scheduler = CosineAnnealingLR(self.opt, T_max=main_steps)
            self.scheduler = SequentialLR(self.opt, schedulers=[warm_up_scheduler, cosine_scheduler], milestones=[warm_up_steps])

            criterion = nn.BCELoss()

            for epoch in range(num_epochs):
                loss = 0
                batch_shuffled_idx = np.random.permutation(samples.shape[0])
                for i in tqdm(range(interval)):
                    start_pt = i * batch_size
                    end_pt = min((i + 1) * batch_size, samples.shape[0])
                    local_idx = batch_shuffled_idx[start_pt:end_pt]
                    batch_sample = samples[local_idx, ...]
                    batch_label = labels[local_idx, ...]

                    predictions = torch.squeeze(self.forward(batch_sample), dim=-1)
                    loss = criterion(predictions, batch_label)

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    self.scheduler.step()
                    
                    print("loss:", loss.item())
                    print("lr:", self.opt.param_groups[0]['lr'])

            if save_dir is not None:
                tmp_save_dir= Path(save_dir)
                tmp_save_dir.mkdir(parents=True, exist_ok=True)
                model_file = tmp_save_dir / 'ps_model.pth'
                self.save_model(model_file)

        else:
            self.load_model(load_dir)

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))



if __name__ == '__main__':

    #Model Training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_id = np.loadtxt('scholar_id.csv', delimiter=',', dtype=str).ravel()
    train_data = get_index(train_id)
    train_data = np.nan_to_num(np.array(train_data), nan=0, posinf=1e6, neginf=0)
    train_data = np.transpose(train_data, (1, 2, 0))
    train_data_feature = feature_extra(train_data)

    scholar_id_path = "Data/Shenzhen_Scholar_ge7"
    all_id = np.array([f.name for f in os.scandir(scholar_id_path) if f.is_dir()])

    contrast_all_id = np.setdiff1d(all_id, train_id)
    contrast_id = contrast_all_id[np.random.choice(len(contrast_all_id), size=4*len(train_id), replace=False)]
    contrast_data = get_index(contrast_id)
    contrast_data =  np.nan_to_num(np.array(contrast_data), nan=0, posinf=1e6, neginf=0)
    contrast_data = np.transpose(contrast_data, (1, 2, 0))
    contrast_data_feature = feature_extra(contrast_data)

    similarity = compute_similar(contrast_data_feature, train_data_feature)
    similarity = 0.75 * (similarity - np.min(similarity)) / (np.max(similarity) - np.min(similarity))

    datas = torch.cat([torch.tensor(train_data, device=device), torch.tensor(contrast_data, device=device)], dim=0)
    labels = torch.cat([
        torch.ones(len(train_data), device=device, dtype=torch.float32),
        torch.tensor(similarity, device=device, dtype=torch.float32)
    ], dim=0)

    PS_estimation = PS_Estimation(time_length=11, feature_dim=4, hidden_dim=32, device=device)
    # PS_estimation.fit_data(samples=datas, labels=labels, save_dir='saved_checkpoint', batch_size=24, num_epochs=200, warm_up_epochs=20, lr=5e-5)
    PS_estimation.fit_data(samples=datas, labels=labels, load_dir='saved_checkpoint/ps_model.pth')

    #Case Study
    train_id = '24824532900'
    contrast_id = ['56421165300', '56423261700', '56393916300', '56411251400', '56580731300', '56493248300', '56488693800', '56525773200', '56381779900','56163856700', \
                   '56176657400', '56161393800', '56102445200', '56118040000', '57075349600', '56982091400', '56942883900', '57187639800','36480735300', '57189443191']
    contrast_data = get_index(contrast_id)
    contrast_data =  np.nan_to_num(np.array(contrast_data), nan=0, posinf=1e6, neginf=0)
    contrast_data = np.transpose(contrast_data, (1, 2, 0))
    contrast_data = torch.tensor(contrast_data, device=device, dtype=torch.float32)
    ps_score = PS_estimation(contrast_data).detach().cpu().numpy()
    ps_score = ps_score.flatten()
    indices = np.argsort(ps_score)[-15:][::-1]
    indices = indices.flatten().astype(int)
    contrast_id = [contrast_id[i] for i in indices]
    ps_score = ps_score[indices]

    _type_ = 'co_authors' #'co_authors', 'paper_num', 'Sum_IF'
    did_year = '2020'
    year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    new_df = train_and_contrast(train_id, contrast_id, did_year, _type_)
    new_df.to_csv(f'./{_type_}_contrast_final.csv')
    
    y = new_df[_type_].to_list()
    n = [train_id] + contrast_id
    y_2d = [y[i*11:(i+1)*11] for i in range(len(n))]

    mode = int(input("请输入1，2，3："))
    if mode == 1:
        drawing_all_data(y_2d, year, did_year, n, save_path=f'image/{_type_}/all_data.png')
    elif mode == 2:
        results = DID(new_df, ps_score,  _type_, f'./{_type_}_contrast_final.csv')   
        draw_parallel(results, did_year, save_path=f'image/{_type_}/DID.png')
    
    else:
        results = DID_after(new_df, ps_score, _type_, f'./{_type_}_contrast_final.csv')
        print(results.summary())