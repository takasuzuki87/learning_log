
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import seaborn as sns
import glob


# ## DF一覧
# * df1・・・1回目の受講データ。
# * df_tot・・・全ての受講データ。
# * df1org・・・7/3からいるメンバーの1回目受講データ。
# * df_totorg・・・7/3からいるメンバーの全ての受講データ。
# * df1_orgm・・・7/3からいる【必須】メンバーの1回目受講データ。
# * df1_orga・・・7/3からいる【任意】メンバーの1回目受講データ。
# * df_totorgm・・・7/3からいる【必須】メンバーの全ての受講データ。
# * df_totorga・・・7/3からいる【任意】メンバーの全ての受講データ。

# In[2]:


# 7月の受講者別カード単位詳細を取り込む
csv_files=glob.glob('??_UserCoursedtl.csv')
csv_files


# In[3]:


# CSVファイルを統合する
#https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
frame = pd.DataFrame()
list_ = []
for file_ in csv_files:
    data = pd.read_csv(file_,index_col=None, header=1)
    list_.append(data)
df = pd.concat(list_)
df.head()


# In[4]:
df.columns

# In[5]:
# NaNを前と同じ値になるように修正
df.fillna(method='ffill', inplace = True)

# In[ ]:
columns = ['ログインID','所属組織', '所属グループ', 'カードコード','カード名',  'カード種別',
           '表示開始日時','受講回（回目）','学習開始日時','学習終了日時','学習時間']

# 1回目の受講に絞り、事務局の人を外す
first = df['受講回（回目）'] ==1
non_adm = df['所属組織'] !='事務局'

# 1回目の受講のDataframe
df1=df.loc[:, columns][(first)&(non_adm)]
# 全ての学習を含めたDataframe
df_tot=df.loc[:, columns][(non_adm)]

# columns の名称変更
df1.columns = ['LOGINID', 'comp','group', 'c_code','c_name', 'c_type','release','times','start','end','st_time']
df_tot.columns = ['LOGINID', 'comp','group', 'c_code','c_name','c_type','release','times','start','end','st_time']


# In[ ]:
# datetimeの設定
df1['end'] = pd.to_datetime(df1['end'])
df1['start'] = pd.to_datetime(df1['start'])
df1['st_time'] = pd.to_datetime(df1['st_time'])
df1['st_time'] = df1['st_time'].dt.hour*3600 + df1['st_time'].dt.minute*60 + df1['st_time'].dt.second
df1['st_time'] = pd.to_numeric(df1['st_time'])
df1['release'] = pd.to_datetime(df1['release'])

df_tot['end'] = pd.to_datetime(df_tot['end'])
df_tot['start'] = pd.to_datetime(df_tot['start'])
df_tot['st_time'] = pd.to_datetime(df_tot['st_time'])
df_tot['st_time'] = df_tot['st_time'].dt.hour*3600 + df_tot['st_time'].dt.minute*60 + df_tot['st_time'].dt.second
df_tot['st_time'] = pd.to_numeric(df_tot['st_time'])
df_tot['release'] = pd.to_datetime(df_tot['release'])

# 並び替え
df1 = df1[['LOGINID','comp','group', 'c_code','c_name', 'c_type','times','release','start','end','st_time']]
df_tot = df_tot[['LOGINID', 'comp','group','c_code','c_name', 'c_type', 'times','release','start','end','st_time']]


# In[ ]:
# リリースからカード修了までの時間
df1['rel_end_days'] = df1['end'] - df1['release']
df_tot['rel_end_days'] = df_tot['end'] - df_tot['release']

df1['rel_end_days']=df1['rel_end_days'].dt.days
df_tot['rel_end_days']=df_tot['rel_end_days'].dt.days


# In[9]:
# release を日付だけにする
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.dt.date.html
df1['release'] = df1['release'].dt.date
df_tot['release'] = df_tot['release'].dt.date


# In[10]:
# dtypeをdatetimeに戻す
df1['release']=pd.to_datetime(df1['release'])
df_tot['release']=pd.to_datetime(df_tot['release'])


# In[11]:
# 7/3からいるメンバーを取ってくる
df_org = pd.read_csv('20170703_original_1.csv')

# columns の名称変更
df_org.columns = ['LOGINID', 'status']

# df1とdf_orgのマージ
# データのタイプをstringに統一
df_org['LOGINID'] = df_org['LOGINID'].astype(str)
df1['LOGINID'] = df1['LOGINID'].astype(str)
df_tot['LOGINID'] = df_tot['LOGINID'].astype(str)

#タイプの合致確認用
#print(df1.dtypes)
#print(df_org.dtypes)

# df1orgは7/3からいるメンバーの１回目の受講のDataFrame
df1org = pd.merge(df1,df_org, how='left')
# df_totorgは7/3からいるメンバーの全ての受講のDataFrame
df_totorg = pd.merge(df_tot,df_org, how='left')


# In[12]:
# df1orgとdf_totorgについて、必須と任意で分ける
org_mand = df1org['status'] == 'm0703'
org_arb = df1org['status'] == '703'

# df1_orgmは、7/3からいる【必須】メンバーの１回目の受講のDataFrame
df1_orgm = df1org.loc[org_mand]
# df1_orgaは、7/3からいる【任意】メンバーの１回目の受講のDataFrame
df1_orga = df1org.loc[org_arb]

org_tmand = df_totorg['status'] == 'm0703'
org_tarb = df_totorg['status'] == '703'

# df_totorgmは7/3からいる【必須】メンバーの全ての受講のDataFrame
df_totorgm = df_totorg.loc[org_tmand]

# df_totorgaは7/3からいる【任意】メンバーの全ての受講のDataFrame
df_totorga = df_totorg.loc[org_tarb]


# # ここから分析

# **やりたいこと：各release日について、各ユーザーのst_time等を集計する**
#
# 1週間以内に終えている人の学習状況（学習カード枚数…復習度合い、学習時間）

# ## release日毎の分析
# 表を作っていく

# In[15]:
# 各release日の【必須】学習者の母数（7/3からいるメンバー）
group1 = df1_orgm.groupby(['release','LOGINID']).size().groupby(level=0).agg({'total_id':'size'})
group1


# ## 各release日の最後のカードを抽出してDataFrameを作成
# - 修了人数を求める
# - 修了日数の集計

# In[16]:
# 各releaseの最後のカードのSeriesを挙げる
maxcard = df1.groupby('release')['c_code'].max()
dfmax = pd.DataFrame(maxcard)

# DataFrame上のデータを変更（12/4, 12/11分の最後のカードが間違って取れていたため）
dfmax.iloc[20,:] = 141223
dfmax.iloc[21,:] = 144172

# ただし、7月の最初の方はc_codeの最大が最後のカードとは限らないので、修正を加える。
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.replace.html
dfmax_r=dfmax['c_code'].replace(to_replace=[111093,111095], value=[108264,108273])

#DataFrameをListに変換
maxcardlist = dfmax_r.tolist()
#maxcardlist


# In[17]:
# df1_orgmから、各releaseの最後のカードだけを拾ったDataFrameを作る: df1_orgm_last
df1_orgm_last = df1_orgm[df1_orgm['c_code'].isin(maxcardlist)]

# df1_orgm_lastを使ったrelease毎の修了人数
group2 = df1_orgm_last.groupby(['release','LOGINID']).size().groupby(level=0).agg({'finish_id':'size'})

# group1とgroup2の統合
df_release1=group1.join(group2)


# ** df1_orgm_lastのDataFrame**

# In[18]:
#df1_orgm_last.head()


# ### 7日以内に修了した人を集計

# In[19]:
# リリースから修了までの時間が7日以内の人を抽出
sev = df1_orgm_last['rel_end_days'].apply(lambda x: x<8)

#　人数を集計
fin_sev = pd.DataFrame(df1_orgm_last[sev].groupby(['release']).count()['LOGINID'])
# column名の変更
fin_sev=fin_sev.rename(columns = {'LOGINID':'fin_sev'})
# df_releaseに追加
df_release2=df_release1.join(fin_sev)


# In[20]:
# 新しいcolumnの作成：7日以内に終えた人の割合
df_release2['fin_sev_p'] = df_release2['fin_sev']*100/df_release2['total_id']


# In[21]:
df_release2['fin_p'] = df_release2['finish_id']*100/df_release2['total_id']
df_release2


# ### 各release日の修了日数の中央値や平均学習時間を取る
# 学習時間が長いほど修了日数も長い？

# In[22]:
fin_mean = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].mean())
fin_mean = fin_mean.rename(columns = {'rel_end_days':'mean_days'})

fin_med = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].median())
fin_med = fin_med.rename(columns = {'rel_end_days':'med_days'})

fin_75 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.75))
fin_75 = fin_75.rename(columns = {'rel_end_days':'75th_days'})


# In[23]:
df_release3 = df_release2.join(fin_mean).join(fin_med).join(fin_75)
df_release3


# In[24]:
# 各release日のログインID毎の初回学習時間を集計
st1=pd.DataFrame(df1_orgm.groupby(['release','LOGINID'])['st_time'].sum())

# indexを直し、release日毎のmean, median学習時間を集計
st1 = st1.reset_index(level=['LOGINID', 'release'])

stmean = pd.DataFrame(st1.groupby('release')['st_time'].mean())
stmean = stmean.rename(columns = {'st_time':'st_mean'})
stmed = pd.DataFrame(st1.groupby('release')['st_time'].median())
stmed = stmed.rename(columns = {'st_time':'st_med'})


# In[25]:
df_release4 = df_release3.join(stmean).join(stmed)
df_release4


# In[26]:
sns.lmplot(x= 'mean_days', y='st_mean', data=df_release4)


# In[30]:
sns.lmplot(x= 'st_med', y='75th_days', data=df_release4, ci=None)
plt.title('Finishing Days (75th%) and Median Study Time')
plt.xlabel('Median study time')
plt.ylabel('Finishing Days (75th%)')
plt.show()


# ### 修了日数のパーセンタイルの分布

# In[56]:
fin_60 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.60))
fin_65 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.65))
fin_70 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.70))
fin_75 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.75))
fin_80 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.80))
fin_85 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.85))
fin_90 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.90))
fin_95 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.95))
fin_99 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.99))

fin_60 = fin_60.rename(columns = {'rel_end_days':'60th'})
fin_65 = fin_65.rename(columns = {'rel_end_days':'65th'})
fin_70 = fin_70.rename(columns = {'rel_end_days':'70th'})
fin_75 = fin_75.rename(columns = {'rel_end_days':'75th'})
fin_80 = fin_80.rename(columns = {'rel_end_days':'80th'})
fin_85 = fin_85.rename(columns = {'rel_end_days':'85th'})
fin_90 = fin_90.rename(columns = {'rel_end_days':'90th'})
fin_95 = fin_95.rename(columns = {'rel_end_days':'95th'})
fin_99 = fin_99.rename(columns = {'rel_end_days':'99th'})


# In[57]:
fin_p = fin_60.join(fin_65).join(fin_70). join(fin_75).join(fin_80).join(fin_85).join(fin_90).join(fin_95).join(fin_99)
fin_p.head()


# In[58]:
fin_p


# In[59]:
plt.plot(fin_p.index, fin_p['60th'], label='60th')
plt.plot(fin_p.index, fin_p['65th'], label='65th')
plt.plot(fin_p.index, fin_p['70th'], label='70th')
plt.plot(fin_p.index, fin_p['75th'], label='75th')
plt.plot(fin_p.index, fin_p['80th'], label='80th')
plt.plot(fin_p.index, fin_p['85th'], label='85th')
plt.plot(fin_p.index, fin_p['90th'], label='90th')
plt.plot(fin_p.index, fin_p['95th'], label='95th')
#plt.plot(fin_p.index, fin_p['99th'], label='99th')

plt.ylim((0,60))
plt.legend(bbox_to_anchor=(1.25,1))
plt.ylabel('days since release')
plt.show()


# ### 同じことを、任意の人に対して見てみる

# In[60]:
# 各releaseの学習者の母数
group1a = df1_orga.groupby(['release','LOGINID']).size().groupby(level=0).agg({'total_id':'size'})


# In[61]:
# df1_orgaから、各releaseの最後のカードだけを拾ったDataFrameを作る: df1_orgm_last
df1_orga_last = df1_orga[df1_orga['c_code'].isin(maxcardlist)]

# df1_orga_lastを使ったrelease毎の修了人数
group2a = df1_orga_last.groupby(['release','LOGINID']).size().groupby(level=0).agg({'finish_id':'size'})

# group1とgroup2の統合
df_release1a=group1a.join(group2a)


# In[62]:
# リリースから修了までの時間が7日以内の人を抽出
seva = df1_orga_last['rel_end_days'].apply(lambda x: x<8)

#　人数を集計
fin_seva = pd.DataFrame(df1_orga_last[seva].groupby(['release']).count()['LOGINID'])
# column名の変更
fin_seva=fin_seva.rename(columns = {'LOGINID':'fin_sev'})
# df_releaseに追加
df_release2a=df_release1a.join(fin_seva)
# 新しいcolumnの作成：7日以内に終えた人の割合
df_release2a['fin_sev_p'] = df_release2a['fin_sev']*100/df_release2a['total_id']


# In[63]:
df_release2a['fin_p'] = df_release2a['finish_id']*100/df_release2a['total_id']
df_release2a


# In[64]:
#各release日の修了率の棒グラフ
plt.plot(df_release2a.index, df_release2a['fin_p'], label='voluntary')
plt.plot(df_release2.index, df_release2['fin_p'], label='mandatory', color='r')

plt.title('Percentage of finished students')
plt.ylim((90,101))
plt.legend(loc=4)
plt.show()


# In[65]:
#各release日の7日以内修了率の棒グラフ
plt.plot(df_release2a.index, df_release2a['fin_sev_p'], label='voluntary')
plt.plot(df_release2.index, df_release2['fin_sev_p'], label='mandatory', color='r')

plt.title('Percentage of students finishing within 7 days')
plt.ylim((35,75))
plt.legend(loc=2)
plt.show()


# In[66]:
df_release2_m=df_release2.resample("M", how = 'mean')
df_release2a_m=df_release2a.resample("M", how = 'mean')


# In[67]:
#各release日の7日以内修了率の棒グラフ（月次ベース）
plt.plot(df_release2a_m.index, df_release2a_m['fin_sev_p'], label='voluntary (m)', ls = '--', color = 'b', marker = '.')
plt.plot(df_release2_m.index, df_release2_m['fin_sev_p'], label='mandatory (m)', ls = '--', color= 'r', marker = '.')

plt.title('Percentage of students finishing within 7 days (monthly)')
plt.ylim((35,75))
plt.legend(loc=2)
plt.show()


# In[68]:
plt.bar(df_release2a.index, df_release2a['finish_id'])
#plt.bar(df_release2a.index, df_release2a['total_id'])

plt.show()
# - 7日以内修了率は増加傾向にある。
# - 必須、任意とも傾向は似ている。

# ## 1週間以内に修了している人の学習状況

# In[69]:
# LOGINID, release毎に7日以内に終えたかどうかを判定
# Add a boolean column to a DataFrame.
# https://stackoverflow.com/questions/30912403/appending-boolean-column-in-panda-dataframe
df1_orgm_last['fin_sev'] = sev


# In[70]:
# 必要なcolumnsだけに絞る（LOGINID, release, fin_sev）
df1_orgm_last2 = df1_orgm_last[['LOGINID', 'release', 'fin_sev']].set_index(['release', 'LOGINID'])
df1_orgm_last2.head()


# ### 学習時間の集計

# In[71]:
# 各release日のログインID毎の初回学習時間を集計
st1=df1_orgm.groupby(['release','LOGINID'])['st_time'].sum()

# Series to DataFrame
# https://stackoverflow.com/questions/26097916/python-best-way-to-convert-a-pandas-series-into-a-pandas-dataframe
df1_st = pd.DataFrame(data=st1, columns=['st_time']).rename(columns = {'st_time':'st_time_1'})


# In[72]:
df_orgm_rel = df1_orgm_last2.join(df1_st)
df_orgm_rel.head()


# In[73]:
# 各release日のログインID毎の合計学習時間を集計
st_t=df_totorgm.groupby(['release','LOGINID'])['st_time'].sum()
# Series to DataFrame
# https://stackoverflow.com/questions/26097916/python-best-way-to-convert-a-pandas-series-into-a-pandas-dataframe
dftot_st = pd.DataFrame(data=st_t, columns=['st_time']).rename(columns = {'st_time':'st_time_tot'})


# In[74]:
#復習時間を計算
df_orgm_rel2=df_orgm_rel.join(dftot_st)
df_orgm_rel2['st_time_rep'] = df_orgm_rel2['st_time_tot'] - df_orgm_rel2['st_time_1']
#df_orgm_rel2.head()


# In[75]:
# Resetting index
# https://stackoverflow.com/questions/20461165/how-to-convert-pandas-index-in-a-dataframe-to-a-column
df_orgm_rel3 = df_orgm_rel2.reset_index(level='LOGINID')
df_orgm_rel3.head()


# In[76]:
# release日毎に7日以内に修了別の初回学習時間
sns.boxplot(x=df_orgm_rel3.index, y='st_time_1', data=df_orgm_rel3, hue='fin_sev')
plt.legend(loc='upper right')
plt.ylim(0,3600)
plt.xticks(rotation=90)
plt.show()


# In[77]:
# release日毎に7日以内に修了別の合計学習時間
sns.boxplot(x=df_orgm_rel3.index, y='st_time_tot', data=df_orgm_rel3, hue='fin_sev')
plt.legend(loc='upper right')
plt.xticks(rotation=90)
plt.ylim(0,3600)
plt.show()


# In[78]:
df_orgm_fin_sev=df_orgm_rel3.loc[df_orgm_rel3['fin_sev']]
df_orgm_fin_sev.head()


# In[79]:
# LOGINID毎に７日以内に修了した回数を数える
df_orgm_fin_sev.groupby('LOGINID')['fin_sev'].count()


# In[80]:
plt.hist(df_orgm_fin_sev.groupby('LOGINID')['fin_sev'].count(),bins=23)
plt.show()


# ### 学習カード枚数の集計

# In[81]:
sc_t=df_totorgm.groupby(['release','LOGINID'])['c_code'].count()
dftot_sc = pd.DataFrame(data=sc_t, columns=['c_code'])
dftot_sc=dftot_sc.rename(columns = {'c_code':'st_cards'})
dftot_sc.head()


# In[82]:
df_orgm_rel4=df_orgm_rel2.join(dftot_sc)
df_orgm_rel4 = df_orgm_rel4.reset_index(level='LOGINID')
df_orgm_rel4.head()


# - df_orgm_rel4を使って、各release日、７日以内修了別に学習カードの平均枚数を集計

# In[83]:
df_orgm_r_fs1 = pd.DataFrame(df_orgm_rel4.groupby([df_orgm_rel4.index,'fin_sev'])['st_cards'].mean())
df_orgm_r_fs2 = pd.DataFrame(df_orgm_rel4.groupby([df_orgm_rel4.index,'fin_sev'])['st_time_1'].mean())
df_orgm_r_fs3 = pd.DataFrame(df_orgm_rel4.groupby([df_orgm_rel4.index,'fin_sev'])['st_time_tot'].mean())
df_orgm_r_fs4 = pd.DataFrame(df_orgm_rel4.groupby([df_orgm_rel4.index,'fin_sev'])['st_time_tot'].median())
df_orgm_r_fs4=df_orgm_r_fs4.rename(columns = {'st_time_tot':'st_time_tot_median'})
df_orgm_r_fs5 = pd.DataFrame(df_orgm_rel4.groupby([df_orgm_rel4.index,'fin_sev'])['st_time_rep'].mean())
df_orgm_r_fs6 = pd.DataFrame(df_orgm_rel4.groupby([df_orgm_rel4.index,'fin_sev'])['st_time_1'].median())
df_orgm_r_fs6=df_orgm_r_fs6.rename(columns = {'st_time_1':'st_time_1_median'})
df_orgm_r_fs=df_orgm_r_fs3.join(df_orgm_r_fs2).join(df_orgm_r_fs6).join(df_orgm_r_fs4).join(df_orgm_r_fs5).join(df_orgm_r_fs1)
df_orgm_r_fs=df_orgm_r_fs.rename(columns = {'st_time_tot':'st_time_tot_mean', 'st_time_1':'st_time_1_mean',
                                        'st_cards':'st_cards_mean', 'st_time_rep':'st_time_rep_mean'})
df_orgm_r_fs


# In[84]:
# 各release日、７日以内修了別に学習カードの平均枚数の棒グラフ
# Plotting with multi-index: https://stackoverflow.com/questions/25386870/pandas-plotting-with-multi-index
df_orgm_r_fs['st_cards_mean'].unstack(level=1).plot(kind='bar')
plt.show()

# 毎週７日以内に修了している人のほうが学習カードが多い。→復習をしているということであり、効果も高いと考えられる？
# 実際復習している人の割合はどのくらいなのだろう？


# In[85]:
# 各release日、７日以内修了別に学習カードの平均1回目学習時間の棒グラフ
# Plotting with multi-index: https://stackoverflow.com/questions/25386870/pandas-plotting-with-multi-index
df_orgm_r_fs['st_time_1_mean'].unstack(level=1).plot(kind='bar')
plt.title("Mean studying time (first time)")
plt.ylabel('study time (sec)')
plt.show()


# In[86]:
# 各release日、７日以内修了別に学習カードの1回目学習時間の中央値棒グラフ
# Plotting with multi-index: https://stackoverflow.com/questions/25386870/pandas-plotting-with-multi-index
df_orgm_r_fs['st_time_1_median'].unstack(level=1).plot(kind='bar')
plt.title("Median studying time (first time)")
plt.ylabel('study time (sec)')
plt.show()


# In[87]:
# 各release日、７日以内修了別に学習カードの平均合計学習時間の棒グラフ
# Plotting with multi-index: https://stackoverflow.com/questions/25386870/pandas-plotting-with-multi-index
df_orgm_r_fs['st_time_tot_mean'].unstack(level=1).plot(kind='bar')
plt.title("Mean studying time")
plt.ylabel('study time')
plt.show()


# In[88]:
# 各release日、７日以内修了別に学習カードの合計学習時間の中央値の棒グラフ
# Plotting with multi-index: https://stackoverflow.com/questions/25386870/pandas-plotting-with-multi-index
df_orgm_r_fs['st_time_tot_median'].unstack(level=1).plot(kind='bar')
plt.title("Median studying time")
plt.ylabel('study time')
plt.show()

# 学習時間が短い回は７日以内修了率も高いのでは？
# 学習時間が短い回は学習枚数多い？


# In[89]:
# 各release日、７日以内修了別に学習カードの平均復習時間の棒グラフ
# Plotting with multi-index: https://stackoverflow.com/questions/25386870/pandas-plotting-with-multi-index
df_orgm_r_fs['st_time_rep_mean'].unstack(level=1).plot(kind='bar')
plt.show()


# In[90]:
# １回目の学習時間と復習時間の相関
df_temp = df_orgm_r_fs.reset_index(level='fin_sev')
sns.lmplot(x= 'st_time_1_mean', y='st_time_rep_mean', data=df_temp, hue='fin_sev')
plt.show()


# In[48]:
dates = df1_orgm_last['release']
labels = dates.strftime('%b %d')


# In[91]:
# relaese毎の修了日数boxplot
sns.boxplot(x='release', y='rel_end_days', data=df1_orgm_last)
plt.xticks(rotation=90)
plt.title('Boxplot of finishing duration')
plt.ylim((-5,60))
plt.show()


# ### カードタイプ別の学習時間の変化

# In[13]:
df_orgm_ct = pd.DataFrame(df1_orgm.groupby(['release','c_type'])['st_time'].mean())
df_orgm_ct


# In[53]:
df1_orgm_last[df1_orgm_last['release']=='2017-08-28']['rel_end_days'].min()


# In[81]:
# 3600秒以上をカット
lesshour = df_st2['st_time'] < 3600
df_st3 = df_st2[(lesshour)]


# In[91]:
# Resetting index of df_st
# https://stackoverflow.com/questions/20461165/how-to-convert-pandas-index-in-a-dataframe-to-a-column
df_st2 = df_st.reset_index(level='LOGINID')
df_st2.head()
