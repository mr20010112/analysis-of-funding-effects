import pandas as pd
import os
import numpy as np

from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
'''
scopus_id_author_dict = {
        '24824532900': '胡战利@中科院深圳先进院',
        '56128400700': '王荣刚@北京大学深圳国际研究生院',
        '8580719000': '李景治@南方科技大学',
    }
'''
scopus_id_author_dict = {
        '55821788500': '李子刚@深圳湾实验室',
        '37041812700': '曾泽兵@湖南大学',
        '56430078900': '张立源@南方科技大学',
        '7404115796': '喻学锋@中国科学院深圳先进技术研究院',
        '7402202127': '袁小聪@深圳大学',
        '34770996100': '贺彦兵@清华大学深圳国际研究生院',
        '14038527700': '徐文福@哈尔滨工业大学（深圳）',
        '24824532900': '胡战利@中科院深圳先进院',
        '56128400700': '王荣刚@北京大学深圳研究生院',
        '8580719000': '李景治@南方科技大学',
        '55234284300': '李霄鹏@深圳大学',
        '56931738300': '吴松@深圳大学',
        # '56789285500': '张倩@哈尔滨工业大学（深圳）',
        # '55742830000': '梁慧@哈尔滨工业大学（深圳）',
        # '55850164300': '阳江@南方科技大学',
        # '55571321100': '江朝伟@哈尔滨工业大学（深圳）',
        # '57113803900': '朱英杰@中国科学院深圳先进技术研究院',
        # '57104156300': '张纵辉@香港中文大学（深圳）',
        # '35174957200': '刘厚德@清华大学深圳国际研究生院',
    }
scopus_id_prj_name_dict = {
        '55821788500': '针对表观遗传学重要靶标的抗肿瘤多肽调节剂研究',
        '37041812700': '光电磁功能的新型共轭电子材料',
        '56128400700': '超高清视频编码与处理',
        '56430078900': '低温量子输运的研究',
        '7404115796': '二维黑磷',
        '7402202127': '近场自旋光子技术:基础、表征与应用',
        '24824532900': '基于时空核矩阵的高时间分并动态PET/MR定量参数成像研究',
        '34770996100': '室温高性能全固态锂金属电池及其电荷存储和离子输运模型',
        '14038527700': '大型仿生扑翼机器人编队飞行动カ学与协同控制研究',
        '55234284300': '介观配位超分子自组装及应用',
        '8580719000': '电磁正逆散射中的形状分析理论与成像算法',
        '57113803900': '药物成瘾的神经环路机制',
        '56931738300': '膀胱癌分子机理研究及早筛与诊疗体系构建',
    }

def get_all_metrics(scopus_ID):
    df_ZKY = pd.read_csv('./Data/Journal/FQBJCR2022-UTF8.csv', encoding='utf-8')
    df_JCR = pd.read_csv('./Data/Journal/jcr_2021_wos.csv')
    each_key = scopus_ID
    save_dir = './Data/Shenzhen_Scholar_ge7/' + each_key + '/'
    res_dict = {
        '年累计发表论文数': [],
        '年度平均IF': [],
        '年度累计IF': [],
        '年度最高IF': [],
        '年度平均JCI': [],
        '年度累计JCI': [],
        '年度最高JCI': [],
        '年度累计发表JCR分区Q1论文数量': [],
        '年度累计发表JCR分区Q1和Q2论文数量': [],
        '年度累计发表中科院分区一区论文数量': [],
        '年度累计发表中科院分区一区和二区论文数量': [],
        '年度累计发表中科院分区顶级期刊论文数量': [],
        '年度累计合作者数量': [],
    }

    list_of_years = []

    if os.path.exists(save_dir):
        df = pd.read_csv(save_dir + 'publications.csv')
        #start_year = int(min(df['coverDate'].values).split('-')[0])
        start_year = 2012
        end_year = 2023
        #list_of_years = range(start_year, end_year + 1)
        list_of_years = range(start_year, end_year)
        for each_year in list_of_years:
            num_of_documents = 0
            cumulative_IF = 0
            cumulative_JCI = 0
            max_IF = 0
            max_JCI = 0
            JCR_1 = 0
            JCR_2 = 0
            ZKY_1 = 0
            ZKY_2 = 0
            ZKY_TOP = 0
            co_author_list = []
            for each_idx in df.index:
                each_publication = df.iloc[each_idx]
                each_cover_date = each_publication['coverDate']
                each_cover_year = int(each_cover_date.split('-')[0])
                if each_cover_year == each_year:
                    num_of_documents = num_of_documents + 1
                    if not type(each_publication['author_ids']) is float:
                        co_author_list.extend(each_publication['author_ids'].split(';'))
                    issn = each_publication['issn']
                    if not type(issn) is float and isinstance(issn, str):
                        # print(issn)
                        formated_issn = issn[:4] + '-' + issn[4:]
                        # print(formated_issn)
                        JCR_record = df_JCR[df_JCR['issn'] == formated_issn]
                        if not len(JCR_record) == 0:
                            IF = JCR_record['impact_factor_2021'].values[0]
                            JIF_quantile = JCR_record['jif_quartile'].values[0]
                            JCI = JCR_record['jci_2021'].values[0]
                            cumulative_IF = cumulative_IF + IF
                            cumulative_JCI = cumulative_JCI + JCI
                            if max_IF < IF:
                                max_IF = IF
                            if max_JCI < JCI:
                                max_JCI = JCI
                            if type(JIF_quantile) is not float:
                                if 'Q1' in JIF_quantile:
                                    JCR_1 = JCR_1 + 1
                                elif 'Q2' in JIF_quantile:
                                    JCR_2 = JCR_2 + 1
                        ZKY_record = df_ZKY[df_ZKY['ISSN'] == formated_issn]
                        if not len(ZKY_record) == 0:
                            if ZKY_record['大类分区'].values[0] == 1 or ZKY_record['大类分区'].values[0] == '1':
                                ZKY_1 = ZKY_1 + 1
                            elif ZKY_record['大类分区'].values[0] == 2 or ZKY_record['大类分区'].values[0] == '2':
                                ZKY_2 = ZKY_2 + 1
                            if ZKY_record['Top'].values[0] == '是':
                                ZKY_TOP = ZKY_TOP + 1
            num_of_co_authors = len(list(set(co_author_list)))
            if num_of_documents > 0:
                res_dict['年累计发表论文数'].append(num_of_documents)
                res_dict['年度平均IF'].append(cumulative_IF / num_of_documents)
                res_dict['年度累计IF'].append(cumulative_IF)
                res_dict['年度最高IF'].append(max_IF)
                res_dict['年度平均JCI'].append(cumulative_JCI / num_of_documents)
                res_dict['年度累计JCI'].append(cumulative_JCI)
                res_dict['年度最高JCI'].append(max_JCI)
                res_dict['年度累计发表JCR分区Q1论文数量'].append(JCR_1)
                res_dict['年度累计发表中科院分区一区论文数量'].append(ZKY_1)
                res_dict['年度累计发表JCR分区Q1和Q2论文数量'].append(JCR_1 + JCR_2)
                res_dict['年度累计发表中科院分区一区和二区论文数量'].append(ZKY_1 + ZKY_2)
                res_dict['年度累计发表中科院分区顶级期刊论文数量'].append(ZKY_TOP)
                res_dict['年度累计合作者数量'].append(num_of_co_authors)
            else:
                res_dict['年累计发表论文数'].append(0)
                res_dict['年度平均IF'].append(0)
                res_dict['年度累计IF'].append(0)
                res_dict['年度最高IF'].append(0)
                res_dict['年度平均JCI'].append(0)
                res_dict['年度累计JCI'].append(0)
                res_dict['年度最高JCI'].append(0)
                res_dict['年度累计发表JCR分区Q1论文数量'].append(0)
                res_dict['年度累计发表中科院分区一区论文数量'].append(0)
                res_dict['年度累计发表JCR分区Q1和Q2论文数量'].append(0)
                res_dict['年度累计发表中科院分区一区和二区论文数量'].append(0)
                res_dict['年度累计发表中科院分区顶级期刊论文数量'].append(0)
                res_dict['年度累计合作者数量'].append(0)
    return list_of_years, res_dict

def compute_Similar(data, res):

    for i in range(len(data)):
        for j in range(i+1, len(data)):

            curve1 = np.array(data[i])
            curve2 = np.array(data[j])
            similarity = 1 / (1 + np.mean(np.abs(curve1 - curve2)))
            res[i, j] = similarity
            res[j, i] = similarity

def compute_All_MAE():

    paper_num = []
    Sum_JCR_Q1 = []
    Sum_ZKY = []
    Avg_IF = []
    Sum_IF = []
    Max_IF = []
    Sum_JCR = []
    Avg_JCR = []
    Max_JCR = []
    for each_key in scopus_id_author_dict.keys():
        list_of_years, res_dict = get_all_metrics(each_key)
        paper_num.append(res_dict['年累计发表论文数'])
        Sum_JCR_Q1.append(res_dict['年度累计发表JCR分区Q1论文数量'])
        Sum_ZKY.append(res_dict['年度累计发表中科院分区顶级期刊论文数量'])
        Avg_IF.append(res_dict['年度平均IF'])
        Sum_IF.append(res_dict['年度累计IF'])
        Max_IF.append(res_dict['年度最高IF'])
        Sum_JCR.append(res_dict['年度累计JCI'])
        Avg_JCR.append(res_dict['年度平均JCI'])
        Max_JCR.append(res_dict['年度最高JCI'])

    paper_num = [sublist[-12:] for sublist in paper_num]
    Sum_JCR_Q1 = [sublist[-12:] for sublist in Sum_JCR_Q1]
    Sum_ZKY = [sublist[-12:] for sublist in Sum_ZKY]
    Avg_IF = [sublist[-12:] for sublist in Avg_IF]
    Sum_IF = [sublist[-12:] for sublist in Sum_IF]
    Max_IF = [sublist[-12:] for sublist in Max_IF]
    Sum_JCR = [sublist[-12:] for sublist in Sum_JCR]
    Avg_JCR = [sublist[-12:] for sublist in Avg_JCR]
    Max_JCR = [sublist[-12:] for sublist in Max_JCR]

    num_curves = len(paper_num)

    paperNum_mae = np.zeros((num_curves, num_curves))
    Sum_JCR_Q1_mae = np.zeros((num_curves, num_curves))
    Sum_ZKY_mae = np.zeros((num_curves, num_curves))
    Avg_IF_mae = np.zeros((num_curves, num_curves))
    Sum_IF_mae = np.zeros((num_curves, num_curves))
    Max_IF_mae = np.zeros((num_curves, num_curves))
    Sum_JCR_mae = np.zeros((num_curves, num_curves))
    Avg_JCR_mae = np.zeros((num_curves, num_curves))
    Max_JCR_mae = np.zeros((num_curves, num_curves))

    compute_Similar(Sum_JCR_Q1, Sum_JCR_Q1_mae)
    compute_Similar(paper_num, paperNum_mae)
    compute_Similar(Sum_ZKY, Sum_ZKY_mae)
    compute_Similar(Avg_IF,Avg_IF_mae)
    compute_Similar(Sum_IF, Sum_IF_mae)
    compute_Similar(Max_IF, Max_IF_mae)
    compute_Similar(Sum_JCR, Sum_JCR_mae)
    compute_Similar(Avg_JCR, Avg_JCR_mae)
    compute_Similar(Max_JCR, Max_JCR_mae)


    return paperNum_mae, Sum_JCR_Q1_mae, Sum_ZKY_mae, Avg_IF_mae, Sum_IF_mae, Max_IF_mae, Sum_JCR_mae, Avg_JCR_mae, Max_JCR_mae

def draw_clustering_pic(similarity_matrix, title):
    x_labels = []
    y_labels = []

    for keys, values in scopus_id_author_dict.items():
        #x_labels.append(scopus_id_prj_name_dict[keys])
        y_labels.append(scopus_id_prj_name_dict[keys])
    
    plt.rcParams['font.family'] = 'Songti SC'
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    
    #plt.xticks(np.arange(len(similarity_matrix)), x_labels)
    plt.yticks(np.arange(len(similarity_matrix)), y_labels)
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            text = "{:.2f}".format(similarity_matrix[i, j])
            plt.text(j, i, text, ha='center', va='center', color='w')
    

    plt.colorbar()
    plt.title(title)
    plt.xlabel('研究课题')
    plt.ylabel('研究课题')
    plt.show()

if __name__ == "__main__":
    
    paperNum_mae, Sum_JCR_Q1_mae, Sum_ZKY_mae, Avg_IF_mae, Sum_IF_mae, Max_IF_mae, Sum_JCR_mae, Avg_JCR_mae, Max_JCR_mae = compute_All_MAE()
    title =['年累计发表论文数', '年度累计发表JCR分区Q1论文数量', '年度累计发表中科院分区顶级期刊论文数量', '年度平均IF','年度累计IF', '年度最高IF', '年度累计JCI', '年度平均JCI','年度最高JCI']
    draw_clustering_pic(Max_JCR_mae, title[8])

    #list_of_years, res_dict = get_all_metrics('7402178394')
    #print(res_dict)