import time

import streamlit as s
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
#progress
progress = s.progress(0)
for i in range(100):
    time.sleep(0.1)
    progress.progress(i+1)
about = s.sidebar.radio("",['Home','About us'])

if about == "Home":
    s.write(""" # DECISION SUPPORT SYSTEM """)  # title
    s.subheader(""" MODEL PREDICTIONS BASED ALREADY TRINED MODEL""")
    s.subheader(""" steps
                 1. upload file
                 2. File must contain REQUIRED  columns
                 3. select NAN preprocessing according your requiremet
                 4.Download the Pickle from link
                 """)

    # configurations
    #s.set_option('deprecation.showfileUploderEncoding',False)
    #side bar
    s.sidebar.subheader("upload file")

    #upload file
    file = s.sidebar.file_uploader(label="CSV or EXCEL FILE",type=['csv','xlsx'])
    global df
    if file is not None:
        s.write(''' ## Orginal file''')
        try:
            df = pd.read_csv(file)
        except Exception as e:
            df = pd.read_excel(file)
    try:
        s.write(df)
    except:
        s.write('Please upload file in SIDEBAR')



    try:
        if df is not None:
            s.subheader('''Required columns''')
            s.write(''''VARIANT_CLASS', 'TLOD', 'shiftscore', 'Sample.AF', 'SIFT', 'MBQ', 'MFRL', 'MMQ', 'Sample.AD', 'Sample.F1R2',
                 'Sample.F2R1', 'DP', 'GERMQ', 'MPOS',
                 'POPAF', 'Sample.DP''')
            data = df[
                ['VARIANT_CLASS', 'TLOD', 'shiftscore', 'Sample.AF', 'SIFT', 'MBQ', 'MFRL', 'MMQ', 'Sample.AD', 'Sample.F1R2',
                 'Sample.F2R1', 'DP', 'GERMQ', 'MPOS',
                 'POPAF', 'Sample.DP']]
            #s.write(data)
            s.write('''## categorical value conversion''')
            a = {'SNV': 0, 'substitution': 1, 'deletion': 2, 'insertion': 3}
            data['VARIANT_CLASS'] = data['VARIANT_CLASS'].map(a)
            b = {'deleterious': 0, 'tolerated': 1, 'deleterious_low_confidence': 2,
                 'tolerated_low_confidence': 3}
            data['SIFT'] = data['SIFT'].map(b)
        else:
            s.write('Please upload correct file')
        #s.write(data)
    except:
        s.write("")

    # counting nan values
    #s.write("NAN values count")
    try:
        a = data.isna().sum()
        aa = dict(a)
        key_list = list(aa.keys())
        value_list = list(aa.values())
        if len(key_list) == len(value_list):
          for i in range(len(value_list)):
            if value_list[i] >0:
               value_index = i
               null_values = key_list[value_index]
               #print(null_values)
        else:
            s.write("columns are not matching")
        s.write(null_values," ","columns countain NAN values")
    except:
        print("d")
    try:
        s.sidebar.subheader("Select NAN process method")
        NAN = s.sidebar.selectbox("",{"DROP","MEAN"})
        if NAN == "DROP":
            data = data.dropna()
        if NAN=="MEAN":
            mean_null = data[null_values].mean()
            data[null_values].fillna(value=mean_null, inplace=True)
        s.write("Final Shape of Dataset",data.shape)
        s.write('correlation Heatmap')
        fig = plt.figure(figsize=(12, 10))
        cor = data.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        #s.write(fig)

    except:
        s.write("")


    #data vizualization


    # s.sidebar.subheader(""" Select type of graph""")
    # choose = s.sidebar.selectbox("",{'Heatmap','pairplot'})

    # def grsph(g):
    #     try:
    #         if g=='pairplot':
    #
    #             fig = sns.pairplot(data)
    #             s.pyplot(fig)
    #             s.write('PairPlot')
    #             #s.write(fig)
    #         if g=='Heatmap':
    #             s.write('correlation Heatmap')
    #             fig = plt.figure(figsize=(12, 10))
    #             cor = data.corr()
    #             sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    #             s.write(fig)
    #     except:
    #         s.write("")
    # grsph(choose)
    #classification




#     try:
#         s.sidebar.subheader("Pickle file")
#         s.sidebar.write("DOWNLOAD [link](https://drive.google.com/file/d/1lvJinJcoRIwERhgcImoBC5jIy2Cs438F/view?usp=sharing)")
#         pickle_file = s.sidebar.file_uploader(label="Download and Upload")
#         if data is not None:
#                 model_load = pickle.load(pickle_file)
#                 out = model_load.predict(data)
#                 #s.write(out)
#                 s.subheader("PREDICTIONS")
#                 p1= pd.DataFrame(out)
#                 p1_count = p1[0].value_counts()
#                 p1_pd = pd.DataFrame(p1_count)
#                 s.write(p1_pd.rename(columns={0:'count'}))
#                 p = p1[0].value_counts().to_dict()
#                 label = []
#                 sizes = []
#                 for x,y in p.items():
#                     label.append(x)
#                     sizes.append(y)
#                 try:
#                     s.subheader('PIE CHART')
#                     f1 = plt.figure(figsize=(10,30))
#                     plt.pie(sizes, labels=label, autopct="%1.1f%%" )
#                     s.pyplot(f1)

#                     s.subheader('BAR CHART')
#                     f2= plt.figure(figsize=(10,10))
#                     plt.bar(label,sizes)
#                     plt.xlabel("cancer type")
#                     plt.ylabel("Frequency")
#                     plt.yticks(rotation=60)
#                     plt.xticks(rotation=70)
#                     plt.show()
#                     s.pyplot(f2)
#                 except:
#                     s.write("error with graphs")
#         else:
#             s.write("error with data")
#     except:
#         s.write("EROORRR")

with open('file/model.pkl','rb') as f:
    model_load = pickle.load(pickle_file)
out = model_load.predict(data)
s.write(out)
if about=='About us':
    s.write("""# DECISION SUPPORT SYSTEM""")
    s.subheader("HI..!")
    s.write("Decision support system for cancer exome datasets,which helps to predict the percentage probability of cancer type for particular features,"
            "through which one can under go personized treatements based on persentage of prediction and early dignosis of cancer also possible.")
    s.subheader("Creators")
    s.write("1. Satyam suresh raiker")
    s.write("   RVCE, Biotechnology")
    s.write("2. Adithya sabhapathi")
    s.write("   RVCE, Biotechnology")
    s.write("3. Satyam singh")
    s.write("   RVCE, Biotechnology")
