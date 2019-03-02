
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import seaborn as sns
import glob

# ## DataFrame list
# * df1: Data from members' first learning
# * df_tot: Data from members' entire learning
# * df1org: Data from the original members' first learning
# * df_totorg: Data from the original members' entire leaning
# * df1_orgm: Data from the original mandatory members' first learning
# * df1_orga: Data from the original voluntary members' first learning
# * df_totorgm: Data from the original mandatory members' entire learning
# * df_totorga: Data from the original voluntary members' entire learning

# Import data from files ending "_UserCoursedtl.csv"
csv_files=glob.glob('??_UserCoursedtl.csv')

# Concatenate files
#https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
frame = pd.DataFrame()
list_ = []
for file_ in csv_files:
    data = pd.read_csv(file_,index_col=None, header=1)
    list_.append(data)
df = pd.concat(list_)
df.head()

# Fill "NaN" to have the same value as the previous row
df.fillna(method='ffill', inplace = True)

# Select columns (in Japanese)
columns = ['ログインID','所属組織', '所属グループ', 'カードコード','カード名',  'カード種別',
           '表示開始日時','受講回（回目）','学習開始日時','学習終了日時','学習時間']

# Select the rows of learning for the first time
first = df['受講回（回目）'] ==1

# Select only members of learning
non_adm = df['所属組織'] !='事務局'

# DataFrame of members' first learning
df1=df.loc[:, columns][(first)&(non_adm)]

# DataFrame of members' entire learning
df_tot=df.loc[:, columns][(non_adm)]

# Rename columns
df1.columns = ['LOGINID', 'comp','group', 'c_code','c_name', 'c_type','release','times','start','end','st_time']
df_tot.columns = ['LOGINID', 'comp','group', 'c_code','c_name','c_type','release','times','start','end','st_time']

# Changing data type to datetime
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

# Reorganizing columns
df1 = df1[['LOGINID','comp','group', 'c_code','c_name', 'c_type','times','release','start','end','st_time']]
df_tot = df_tot[['LOGINID', 'comp','group','c_code','c_name', 'c_type', 'times','release','start','end','st_time']]

# Calculate the time from the release to finishing
df1['rel_end_days'] = df1['end'] - df1['release']
df_tot['rel_end_days'] = df_tot['end'] - df_tot['release']

# Transfer units to 'days'
df1['rel_end_days']=df1['rel_end_days'].dt.days
df_tot['rel_end_days']=df_tot['rel_end_days'].dt.days

# Extract the date of the release time
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.dt.date.html
df1['release'] = df1['release'].dt.date
df_tot['release'] = df_tot['release'].dt.date

# Transfer data type to datetime
df1['release']=pd.to_datetime(df1['release'])
df_tot['release']=pd.to_datetime(df_tot['release'])

# DataFrame of  original members (those who studied from the beginning)
df_org = pd.read_csv('20170703_original_1.csv')

# Change columns names
df_org.columns = ['LOGINID', 'status']


# Change the data type to string
df_org['LOGINID'] = df_org['LOGINID'].astype(str)
df1['LOGINID'] = df1['LOGINID'].astype(str)
df_tot['LOGINID'] = df_tot['LOGINID'].astype(str)

# Merge df1 and df_org
# df1org is a DataFrame of original members' first learning
df1org = pd.merge(df1,df_org, how='left')

# df_totorg is a DataFrame of original members' entire learning
df_totorg = pd.merge(df_tot,df_org, how='left')

# Create two groups (mandatory and voluntary) for df1org and df_totorg
org_mand = df1org['status'] == 'm0703'
org_arb = df1org['status'] == '703'

# df1_orgm is a DataFrame of the original mandatory members' first learning
# df1_orga is a DataFrame of the original voluntary members' first learning
df1_orgm = df1org.loc[org_mand]
df1_orga = df1org.loc[org_arb]

# df_totorgm: DataFrame of the original mandatory members' entire learning
# df_totorga: DataFrame of the original voluntary members' entire learning
org_tmand = df_totorg['status'] == 'm0703'
org_tarb = df_totorg['status'] == '703'

df_totorgm = df_totorg.loc[org_tmand]
df_totorga = df_totorg.loc[org_tarb]


# # Start Analysis # #

# # Examine each member's study time (st_time) for each release date.#
# Focus on members who finish each wekkly course within a week

# Check the participating members for each release date
group1 = df1_orgm.groupby(['release','LOGINID']).size().groupby(level=0).agg({'total_id':'size'})

# ## Create a DataFrame by extracting the last card of each release date
# - Count the number of finished members
# - Count the number of days for each member it took to finish

# Create a Series of the last card for each release date
maxcard = df1.groupby('release')['c_code'].max()
dfmax = pd.DataFrame(maxcard)

# Replace the last card for 12/4 and 12/11
dfmax.iloc[20,:] = 141223
dfmax.iloc[21,:] = 144172

# For some weeks in July, the largest value did not reflect the last card, so replace them with the correct card.
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.replace.html
dfmax_r=dfmax['c_code'].replace(to_replace=[111093,111095], value=[108264,108273])

# Transfer DataFrame to List
maxcardlist = dfmax_r.tolist()

# df1_orgm_last: DataFrame extracting the last card from df1_orgm
df1_orgm_last = df1_orgm[df1_orgm['c_code'].isin(maxcardlist)]

# Number of students finished
group2 = df1_orgm_last.groupby(['release','LOGINID']).size().groupby(level=0).agg({'finish_id':'size'})

# Join group1 and group2
df_release1=group1.join(group2)

# ### Count the number of members finishing each week's material in seven days

# Extract members who finished each week's material in seven days
sev = df1_orgm_last['rel_end_days'].apply(lambda x: x<8)

#　Count the number
fin_sev = pd.DataFrame(df1_orgm_last[sev].groupby(['release']).count()['LOGINID'])
# Rename columns
fin_sev=fin_sev.rename(columns = {'LOGINID':'fin_sev'})
# Join fin_sev to df_release1
df_release2=df_release1.join(fin_sev)

# The ratio of those finished in seven days
df_release2['fin_sev_p'] = df_release2['fin_sev']*100/df_release2['total_id']

# The ratio of those finised
df_release2['fin_p'] = df_release2['finish_id']*100/df_release2['total_id']
df_release2


# ### Get the median days to finish and mean studying time
fin_mean = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].mean())
fin_mean = fin_mean.rename(columns = {'rel_end_days':'mean_days'})

fin_med = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].median())
fin_med = fin_med.rename(columns = {'rel_end_days':'med_days'})

fin_75 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.75))
fin_75 = fin_75.rename(columns = {'rel_end_days':'75th_days'})

df_release3 = df_release2.join(fin_mean).join(fin_med).join(fin_75)

# Get the study time for each member to finish each week's material for the first time
st1=pd.DataFrame(df1_orgm.groupby(['release','LOGINID'])['st_time'].sum())

# Reset index and calculate the mean and median study time for each release day
st1 = st1.reset_index(level=['LOGINID', 'release'])

stmean = pd.DataFrame(st1.groupby('release')['st_time'].mean())
stmean = stmean.rename(columns = {'st_time':'st_mean'})
stmed = pd.DataFrame(st1.groupby('release')['st_time'].median())
stmed = stmed.rename(columns = {'st_time':'st_med'})


# Join DataFrame
df_release4 = df_release3.join(stmean).join(stmed)
df_release4


# Plot the mean days to finish and mean study time for each material
sns.lmplot(x= 'mean_days', y='st_mean', data=df_release4)


# Plot the 75th% of finishing days and median study time for each material
sns.lmplot(x= 'st_med', y='75th_days', data=df_release4, ci=None)
plt.title('Finishing Days (75th%) and Median Study Time')
plt.xlabel('Median study time')
plt.ylabel('Finishing Days (75th%)')
plt.show()


# ### Plot the number of days of 60th to 95th percentile for each material

# In[56]:
fin_60 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.60))
fin_65 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.65))
fin_70 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.70))
fin_75 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.75))
fin_80 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.80))
fin_85 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.85))
fin_90 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.90))
fin_95 = pd.DataFrame(df1_orgm_last.groupby('release')['rel_end_days'].quantile(.95))

fin_60 = fin_60.rename(columns = {'rel_end_days':'60th'})
fin_65 = fin_65.rename(columns = {'rel_end_days':'65th'})
fin_70 = fin_70.rename(columns = {'rel_end_days':'70th'})
fin_75 = fin_75.rename(columns = {'rel_end_days':'75th'})
fin_80 = fin_80.rename(columns = {'rel_end_days':'80th'})
fin_85 = fin_85.rename(columns = {'rel_end_days':'85th'})
fin_90 = fin_90.rename(columns = {'rel_end_days':'90th'})
fin_95 = fin_95.rename(columns = {'rel_end_days':'95th'})

fin_p = fin_60.join(fin_65).join(fin_70). join(fin_75).join(fin_80).join(fin_85).join(fin_90).join(fin_95)

plt.plot(fin_p.index, fin_p['60th'], label='60th')
plt.plot(fin_p.index, fin_p['65th'], label='65th')
plt.plot(fin_p.index, fin_p['70th'], label='70th')
plt.plot(fin_p.index, fin_p['75th'], label='75th')
plt.plot(fin_p.index, fin_p['80th'], label='80th')
plt.plot(fin_p.index, fin_p['85th'], label='85th')
plt.plot(fin_p.index, fin_p['90th'], label='90th')
plt.plot(fin_p.index, fin_p['95th'], label='95th')

plt.ylim((0,60))
plt.legend(bbox_to_anchor=(1.25,1))
plt.ylabel('days since release')
plt.show()

# ## How those who finish the weekly material within a week study

# Determine whether a member finished a meterial within a week
# Add a boolean column to a DataFrame.
# https://stackoverflow.com/questions/30912403/appending-boolean-column-in-panda-dataframe
df1_orgm_last['fin_sev'] = sev

# Extract columns LOGINID, release, fin_sev
df1_orgm_last2 = df1_orgm_last[['LOGINID', 'release', 'fin_sev']].set_index(['release', 'LOGINID'])

# ### Sum up the first study time
st1=df1_orgm.groupby(['release','LOGINID'])['st_time'].sum()

# Series to DataFrame
# https://stackoverflow.com/questions/26097916/python-best-way-to-convert-a-pandas-series-into-a-pandas-dataframe
df1_st = pd.DataFrame(data=st1, columns=['st_time']).rename(columns = {'st_time':'st_time_1'})

df_orgm_rel = df1_orgm_last2.join(df1_st)

# Boxplot the first study time for each material; whether finised within a week
sns.boxplot(x=df_orgm_rel3.index, y='st_time_1', data=df_orgm_rel3, hue='fin_sev')
plt.legend(loc='upper right')
plt.ylim(0,3600)
plt.xticks(rotation=90)
plt.show()

# Count the number of times finishing within a week for each member
df_orgm_fin_sev=df_orgm_rel3.loc[df_orgm_rel3['fin_sev']]
df_orgm_fin_sev.groupby('LOGINID')['fin_sev'].count()

# Create a histogram
plt.hist(df_orgm_fin_sev.groupby('LOGINID')['fin_sev'].count(),bins=23)
plt.show()


# ### Counting the number of study cards
sc_t=df_totorgm.groupby(['release','LOGINID'])['c_code'].count()
dftot_sc = pd.DataFrame(data=sc_t, columns=['c_code'])
dftot_sc=dftot_sc.rename(columns = {'c_code':'st_cards'})

df_orgm_rel4=df_orgm_rel2.join(dftot_sc)
df_orgm_rel4 = df_orgm_rel4.reset_index(level='LOGINID')

# Calculate the mean numbers of cards for each material and whether one finished in a week
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

# Bar graph of the mean numbers of cards for each material and whether one finished in a week
# Plotting with multi-index: https://stackoverflow.com/questions/25386870/pandas-plotting-with-multi-index
df_orgm_r_fs['st_cards_mean'].unstack(level=1).plot(kind='bar')
plt.show()

# In[90]:
# Correlation of the first study time and the re-study time
df_temp = df_orgm_r_fs.reset_index(level='fin_sev')
sns.lmplot(x= 'st_time_1_mean', y='st_time_rep_mean', data=df_temp, hue='fin_sev')
plt.show()

# Boxplot of finishing days for each material
sns.boxplot(x='release', y='rel_end_days', data=df1_orgm_last)
plt.xticks(rotation=90)
plt.title('Boxplot of finishing duration')
plt.ylim((-5,60))
plt.show()
