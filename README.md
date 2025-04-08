# 一、筛选对照组

## co-author.py

### 功能
用于学术合作分析，特别是对深圳地区学者的网络进行分析，筛选出与实验组有相似合作者数量的对照组。

### 输入
实验组的Scopus ID

### 输出
对照组的csv文件

### 示例
```python
if __name__ == "__main__":
        
    
    #print(suss_summaries)
    # 输入实验组的Scopus ID
    suss_summaries = test_passing('56128400700')
    suss_summaries.to_csv('./co_authors_contrast.csv')
```

## paper_num.py

### 功能
用于学术合作分析，特别是对深圳地区学者的网络进行分析，筛选出与实验组有相似论文发表数量的对照组。

### 输入
实验组的Scopus ID

### 输出
对照组的csv文件

### 示例
```python
if __name__ == "__main__":
        
    
    #print(suss_summaries)
    # 输入实验组的Scopus ID
    suss_summaries = test_passing('56128400700')
    suss_summaries.to_csv('./paper_num_contrast.csv')
```

## Sum_IF.py

### 功能
用于学术合作分析，特别是对深圳地区学者的网络进行分析，筛选出与实验组有相似学者影响因子的对照组。

### 输入
实验组的Scopus ID

### 输出
对照组的csv文件

### 示例
```python
if __name__ == "__main__":
        
    
    #print(suss_summaries)
    # 输入实验组的Scopus ID
    suss_summaries = test_passing('56128400700')
    suss_summaries.to_csv('./Sum_IF_contrast_new_56128400700.csv')
```

# 二、运行DID分析

## DID_co_authors.py

### 功能
用于学术合作分析，特别是对深圳地区学者的网络进行分析，以及使用DID来评估杰青基金对合作者数量的影响。
- 获取相似学者: 根据特定学者的Scopus ID，从预设的深圳学者网络中找到最相似的L个学者。
- 学者指数获取: 输入一个作者ID列表，计算和返回每个作者的多种学术指标，例如年度累计发表论文数、JCR Q1论文数、中科院顶级期刊论文数、平均和最大影响因子等。
- 差分分析 (DID): 进行DID分析以评估杰青基金对于合作者数量的影响。
- 数据可视化: 绘制不同模式下的数据图表，包括合作者数量随时间的变化以及DID分析结果的可视化。

### 输入
1. **年份列表**: 这是一个包含从2012年到2022年的年份列表，用于时间序列分析。
2. **主要研究者ID (`train_id`)**: 一个Scopus ID，表示主要的研究者。
3. **对比研究者ID列表 (`contrast_id`)**: 一个包含多个Scopus ID的列表，用于与主要研究者进行对比分析。
4. **模式选择**: 用户输入一个数字（1、2或3），用于选择不同的输出模式：
    - **1**: 绘制全部数据的图表。
    - **2**: 进行DID分析并绘制对比结果图。
    - **3**: 进行DID分析的后续处理并打印统计摘要。

### 输出
1. **CSV文件**: 将 `train_and_contrast` 函数处理后的DataFrame保存为CSV文件，此文件包含合作者数量等信息。
2. **图表**:
    - **模式1**: 显示所有研究者在不同年份的合作者数量的变化图。
    - **模式2**: 显示DID分析的结果，通常是一个并行坐标图，用于对比不同年份和研究者之间的差异。
3. **统计摘要**:
    - **模式3**: 打印DID分析后的统计摘要，提供更详细的统计数据，如系数、标准误等。

### 示例

```python
if __name__ == "__main__":
	# 输入研究的年份
    year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
	# 输入实验组的Scopus ID
    train_id = '24824532900'
    # 输入对照组的Scopus ID
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

# 最后在命令框中输入对应的模式
```


## DID_paper_num.py

### 功能
用于学术论文的数量分析，特别是对深圳地区学者的网络进行分析，以及使用DID来评估杰青基金对学术论文数量的影响。
- 获取相似学者: 根据特定学者的Scopus ID，从预设的深圳学者网络中找到最相似的L个学者。
- 学者指数获取: 输入一个作者ID列表，计算和返回每个作者的多种学术指标，例如年度累计发表论文数、JCR Q1论文数、中科院顶级期刊论文数、平均和最大影响因子等。
- 差分分析 (DID): 进行DID分析以评估杰青基金对于学术论文数量的影响。
- 数据可视化: 绘制不同模式下的数据图表，包括合作者数量随时间的变化以及DID分析结果的可视化。

### 输入
1. **年份列表**: 这是一个包含从2012年到2022年的年份列表，用于时间序列分析。
2. **主要研究者ID (`train_id`)**: 一个Scopus ID，表示主要的研究者。
3. **对比研究者ID列表 (`contrast_id`)**: 一个包含多个Scopus ID的列表，用于与主要研究者进行对比分析。
4. **模式选择**: 用户输入一个数字（1、2或3），用于选择不同的输出模式：
    - **1**: 绘制全部数据的图表。
    - **2**: 进行DID分析并绘制对比结果图。
    - **3**: 进行DID分析的后续处理并打印统计摘要。

### 输出
1. **CSV文件**: 将 `train_and_contrast` 函数处理后的DataFrame保存为CSV文件，此文件包含合作者数量等信息。
2. **图表**:
    - **模式1**: 显示所有研究者在不同年份的合作者数量的变化图。
    - **模式2**: 显示DID分析的结果，通常是一个并行坐标图，用于对比不同年份和研究者之间的差异。
3. **统计摘要**:
    - **模式3**: 打印DID分析后的统计摘要，提供更详细的统计数据，如系数、标准误等。

### 示例

```python
if __name__ == "__main__":
    # 输入研究的年份
    year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    # 输入实验组的Scopus ID
    train_id = '35174957200'
    # 输入对照组的Scopus ID
    contrast_id = ['56386293700',
                   '56465472100',
                   '57189622599',
                   '57131657200',
                   '57061428400',
                   '56438243800',
                   '57188990715',
                   '57189588469',
                   '57188590789',
                   '57188830845']
    
    new_df = train_and_contrast(train_id, contrast_id)
    new_df.to_csv('./paper_num_contrast_final.csv')

    y = new_df['paper_num'].to_list()
    n = [train_id] + contrast_id
    y_2d = [y[i*11:(i+1)*11] for i in range(len(n))]

    mode = int(input("请输入1，2，3："))
    if mode == 1:
        drawing_all_data(y_2d, year, n)
    elif mode == 2:
        results = DID(new_df,'./paper_num_contrast_final.csv')
        draw_parallel(results)
    else:
        results = DID_after(new_df,'./paper_num_contrast_final.csv')
        print(results.summary())

# 最后在命令框中输入对应的模式
```

## DID_Sum_IF.py

### 功能
用于学者影响因子的数量分析，特别是对深圳地区学者的网络进行分析，以及使用DID来评估杰青基金对学者影响因子数量的影响。
- 获取相似学者: 根据特定学者的Scopus ID，从预设的深圳学者网络中找到最相似的L个学者。
- 学者指数获取: 输入一个作者ID列表，计算和返回每个作者的多种学术指标，例如年度累计发表论文数、JCR Q1论文数、中科院顶级期刊论文数、平均和最大影响因子等。
- 差分分析 (DID): 进行DID分析以评估杰青基金对于学者影响因子的影响。
- 数据可视化: 绘制不同模式下的数据图表，包括合作者数量随时间的变化以及DID分析结果的可视化。

### 输入
1. **年份列表**: 这是一个包含从2012年到2022年的年份列表，用于时间序列分析。
2. **主要研究者ID (`train_id`)**: 一个Scopus ID，表示主要的研究者。
3. **对比研究者ID列表 (`contrast_id`)**: 一个包含多个Scopus ID的列表，用于与主要研究者进行对比分析。
4. **模式选择**: 用户输入一个数字（1、2或3），用于选择不同的输出模式：
    - **1**: 绘制全部数据的图表。
    - **2**: 进行DID分析并绘制对比结果图。
    - **3**: 进行DID分析的后续处理并打印统计摘要。

### 输出
1. **CSV文件**: 将 `train_and_contrast` 函数处理后的DataFrame保存为CSV文件，此文件包含合作者数量等信息。
2. **图表**:
    - **模式1**: 显示所有研究者在不同年份的合作者数量的变化图。
    - **模式2**: 显示DID分析的结果，通常是一个并行坐标图，用于对比不同年份和研究者之间的差异。
3. **统计摘要**:
    - **模式3**: 打印DID分析后的统计摘要，提供更详细的统计数据，如系数、标准误等。

### 示例

```python
if __name__ == "__main__":
	# 输入研究的年份
    year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    
	# 输入实验组的Scopus ID
    train_id = '35174957200'
    # 输入对照组的Scopus ID
    contrast_id = ['57061465500',
                   '56623638700',
                   '56595881800',
                   '56529943200',
                   '56524685900',
                   '56549486300',
                   '56342866600',
                   '56651449600',
                   '56337432600',
                   '56377984000'
                   ]

    
    new_df = train_and_contrast(train_id, contrast_id)
    new_df.to_csv('./contrast.csv')

    y = new_df['Sum_IF'].to_list()
    n = [train_id] + contrast_id
    y_2d = [y[i*11:(i+1)*11] for i in range(len(n))]

    mode = int(input("请输入1，2，3："))
    if mode == 1:
        drawing_all_data(y_2d, year, n)
    elif mode == 2:
        results = DID(new_df,'./contrast.csv')
        draw_parallel(results)
    
    else:
        results = DID_after(new_df,'./contrast.csv')
        print(results.summary())

# 最后在命令框中输入对应的模式
```