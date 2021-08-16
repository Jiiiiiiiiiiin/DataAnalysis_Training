연습문제
In [1]:
import numpy as np
import pandas as pd
import seaborn as sns
In [2]:
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')
mpg = sns.load_dataset('mpg')
In [9]:
import warnings
warnings.filterwarnings('ignore')
1. Iris
a. 붓꽃 종(species)별로 꽃잎길이(sepal_length), 꽃잎폭(sepal_width), 꽃받침길이(petal_length), 꽃받침폭(petal_width)의 평균, 표준편차 등 기초통계량(describe())을 구하시오.
In [3]:
iris.groupby('species').agg(['mean','std'])
Out[3]:
sepal_length	sepal_width	petal_length	petal_width
mean	std	mean	std	mean	std	mean	std
species								
setosa	5.006	0.352490	3.428	0.379064	1.462	0.173664	0.246	0.105386
versicolor	5.936	0.516171	2.770	0.313798	4.260	0.469911	1.326	0.197753
virginica	6.588	0.635880	2.974	0.322497	5.552	0.551895	2.026	0.274650
b. 이상치를 제거하고 위의 4가지 항목에 대해서 평균, 표준편차를 구하시오.
In [11]:
s = iris[iris.species=='setosa']['sepal_width']
s.mean(), s.std()
Out[11]:
(3.428000000000001, 0.3790643690962886)
In [12]:
q1, q3 = s.quantile(.25), s.quantile(.75)
iqr = q3 - q1
q1, q3, iqr
Out[12]:
(3.2, 3.6750000000000003, 0.4750000000000001)
In [13]:
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
s[(s < lower) | (s > upper)] = np.nan
s.isnull().sum()
Out[13]:
2
In [14]:
s.mean(), s.std()
Out[14]:
(3.4312500000000004, 0.32034306743094015)
In [15]:
def get_new_stat(s):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    s[(s < lower) | (s > upper)] = np.nan
    outlier = s.isnull().sum() != 0
    return outlier, np.round(s.mean(), 2), np.round(s.std(), 4)
In [16]:
species_list, feature_list, mean_list, std_list = [],[],[],[]
outlier_list, new_mean_list, new_std_list = [],[],[]
for species in iris.species.unique():
    for feature in iris.columns[:-1]:
        s = iris[iris.species == species][feature]
        species_list.append(species)
        feature_list.append(feature)
        mean_list.append(np.round(s.mean(), 2))
        std_list.append(np.round(s.std(), 4))

        outlier, new_mean, new_std = get_new_stat(s)
        outlier_list.append(outlier)
        new_mean_list.append(new_mean)
        new_std_list.append(new_std)
In [18]:
df = pd.DataFrame({
    'species': species_list, 'feature': feature_list,
    'mean': mean_list, 'std': std_list,
    'outlier': outlier_list, 'new_mean': new_mean_list, 'new_std': new_std_list
})
df.set_index(['species','feature'], inplace=True)
df
Out[18]:
mean	std	outlier	new_mean	new_std
species	feature					
setosa	sepal_length	5.01	0.3525	False	5.01	0.3525
sepal_width	3.43	0.3791	True	3.43	0.3203
petal_length	1.46	0.1737	True	1.46	0.1291
petal_width	0.25	0.1054	True	0.23	0.0859
versicolor	sepal_length	5.94	0.5162	False	5.94	0.5162
sepal_width	2.77	0.3138	False	2.77	0.3138
petal_length	4.26	0.4699	True	4.29	0.4378
petal_width	1.33	0.1978	False	1.33	0.1978
virginica	sepal_length	6.59	0.6359	True	6.62	0.5935
sepal_width	2.97	0.3225	True	2.96	0.2603
petal_length	5.55	0.5519	False	5.55	0.5519
petal_width	2.03	0.2747	False	2.03	0.2747
2.Titanic
a. 타이타닉호의 승객에 대해 나이와 성별에 의한 카테고리 열인 category1 열을 만드시오. category1 카테고리는 다음과 같이 정의됨
20살이 넘으면 성별을 그대로 사용한다.
20살 미만이면 성별에 관계없이 “child”라고 한다.
In [19]:
titanic['categrory1'] = titanic.apply(lambda r: r.sex if r.age >= 20 else 'child', axis=1)
titanic.tail()
Out[19]:
survived	pclass	sex	age	sibsp	parch	fare	embarked	class	who	adult_male	deck	embark_town	alive	alone	categrory1
886	0	2	male	27.0	0	0	13.00	S	Second	man	True	NaN	Southampton	no	True	male
887	1	1	female	19.0	0	0	30.00	S	First	woman	False	B	Southampton	yes	True	child
888	0	3	female	NaN	1	2	23.45	S	Third	woman	False	NaN	Southampton	no	False	child
889	1	1	male	26.0	0	0	30.00	C	First	man	True	C	Cherbourg	yes	True	male
890	0	3	male	32.0	0	0	7.75	Q	Third	man	True	NaN	Queenstown	no	True	male
b. 타이타닉호의 승객 중 나이를 명시하지 않은 고객은 나이를 명시한 고객의 평균 나이 값이 되도록 titanic 데이터프레임을 고치시오.
In [20]:
titanic['age'] = titanic.age.fillna(titanic.age.mean())
titanic.tail()
Out[20]:
survived	pclass	sex	age	sibsp	parch	fare	embarked	class	who	adult_male	deck	embark_town	alive	alone	categrory1
886	0	2	male	27.000000	0	0	13.00	S	Second	man	True	NaN	Southampton	no	True	male
887	1	1	female	19.000000	0	0	30.00	S	First	woman	False	B	Southampton	yes	True	child
888	0	3	female	29.699118	1	2	23.45	S	Third	woman	False	NaN	Southampton	no	False	child
889	1	1	male	26.000000	0	0	30.00	C	First	man	True	C	Cherbourg	yes	True	male
890	0	3	male	32.000000	0	0	7.75	Q	Third	man	True	NaN	Queenstown	no	True	male
In [21]:
titanic['categrory1'] = titanic.apply(lambda r: r.sex if r.age >= 20 else 'child', axis=1)
titanic.tail()
Out[21]:
survived	pclass	sex	age	sibsp	parch	fare	embarked	class	who	adult_male	deck	embark_town	alive	alone	categrory1
886	0	2	male	27.000000	0	0	13.00	S	Second	man	True	NaN	Southampton	no	True	male
887	1	1	female	19.000000	0	0	30.00	S	First	woman	False	B	Southampton	yes	True	child
888	0	3	female	29.699118	1	2	23.45	S	Third	woman	False	NaN	Southampton	no	False	female
889	1	1	male	26.000000	0	0	30.00	C	First	man	True	C	Cherbourg	yes	True	male
890	0	3	male	32.000000	0	0	7.75	Q	Third	man	True	NaN	Queenstown	no	True	male
c. 성별, 선실(class)별, 출발지(embark_town)별 생존율을 구하시오.
In [22]:
# 'alive': 범주형 --> 'survived': 정수
titanic.groupby('sex')[['survived']].mean()
Out[22]:
survived
sex	
female	0.742038
male	0.188908
In [23]:
titanic.pivot_table('survived', 'class')
Out[23]:
survived
class	
First	0.629630
Second	0.472826
Third	0.242363
In [24]:
titanic.pivot_table('survived', 'embark_town')
Out[24]:
survived
embark_town	
Cherbourg	0.553571
Queenstown	0.389610
Southampton	0.336957
In [27]:
titanic.pivot_table('survived', ['sex', 'class'])
Out[27]:
survived
sex	class	
female	First	0.968085
Second	0.921053
Third	0.500000
male	First	0.368852
Second	0.157407
Third	0.135447
In [25]:
titanic.pivot_table('survived', ['sex', 'class'], 'embark_town')
Out[25]:
embark_town	Cherbourg	Queenstown	Southampton
sex	class			
female	First	0.976744	1.000000	0.958333
Second	1.000000	1.000000	0.910448
Third	0.652174	0.727273	0.375000
male	First	0.404762	0.000000	0.354430
Second	0.200000	0.000000	0.154639
Third	0.232558	0.076923	0.128302
d. 타이타닉호 승객을 ‘미성년자’, ‘청년’, ‘중년’, ‘장년’, ‘노년’ 나이 그룹으로 나누고, 각 그룹별 생존율을 구하시오.
In [28]:
bins = [1, 20, 30, 50, 70, 100]
labels = ["미성년자", "청년", "중년", "장년", "노년"]
titanic['age_cat'] = pd.cut(titanic.age, bins, labels=labels)
titanic.tail()
Out[28]:
survived	pclass	sex	age	sibsp	parch	fare	embarked	class	who	adult_male	deck	embark_town	alive	alone	categrory1	age_cat
886	0	2	male	27.000000	0	0	13.00	S	Second	man	True	NaN	Southampton	no	True	male	청년
887	1	1	female	19.000000	0	0	30.00	S	First	woman	False	B	Southampton	yes	True	child	미성년자
888	0	3	female	29.699118	1	2	23.45	S	Third	woman	False	NaN	Southampton	no	False	female	청년
889	1	1	male	26.000000	0	0	30.00	C	First	man	True	C	Cherbourg	yes	True	male	청년
890	0	3	male	32.000000	0	0	7.75	Q	Third	man	True	NaN	Queenstown	no	True	male	중년
In [29]:
titanic.pivot_table('survived', 'age_cat')
Out[29]:
survived
age_cat	
미성년자	0.424242
청년	0.334152
중년	0.423237
장년	0.355932
노년	0.200000
In [30]:
titanic.pivot_table('survived', ['sex','age_cat'])
Out[30]:
survived
sex	age_cat	
female	미성년자	0.671233
청년	0.723881
중년	0.779070
장년	0.941176
male	미성년자	0.228261
청년	0.142857
중년	0.225806
장년	0.119048
노년	0.200000
e. qcut 명령으로 세 개의 나이 그룹을 만들고, 나이 그룹별 남녀 성비와 생존율을 구하시오.
In [31]:
titanic['age_group'] = pd.qcut(titanic.age, 3, labels=['A1','A2','A3'])
titanic.tail()
Out[31]:
survived	pclass	sex	age	sibsp	parch	fare	embarked	class	who	adult_male	deck	embark_town	alive	alone	categrory1	age_cat	age_group
886	0	2	male	27.000000	0	0	13.00	S	Second	man	True	NaN	Southampton	no	True	male	청년	A2
887	1	1	female	19.000000	0	0	30.00	S	First	woman	False	B	Southampton	yes	True	child	미성년자	A1
888	0	3	female	29.699118	1	2	23.45	S	Third	woman	False	NaN	Southampton	no	False	female	청년	A2
889	1	1	male	26.000000	0	0	30.00	C	First	man	True	C	Cherbourg	yes	True	male	청년	A2
890	0	3	male	32.000000	0	0	7.75	Q	Third	man	True	NaN	Queenstown	no	True	male	중년	A3
In [33]:
titanic.groupby('age_group')[['survived']].mean()
Out[33]:
survived
age_group	
A1	0.411960
A2	0.335526
A3	0.405594
In [34]:
# gender: 남성이면 1, 여성이면 0
titanic['gender'] = titanic.apply(lambda r: 1 if r.sex == 'male' else 0, axis=1)
titanic.tail()
Out[34]:
survived	pclass	sex	age	sibsp	parch	fare	embarked	class	who	adult_male	deck	embark_town	alive	alone	categrory1	age_cat	age_group	gender
886	0	2	male	27.000000	0	0	13.00	S	Second	man	True	NaN	Southampton	no	True	male	청년	A2	1
887	1	1	female	19.000000	0	0	30.00	S	First	woman	False	B	Southampton	yes	True	child	미성년자	A1	0
888	0	3	female	29.699118	1	2	23.45	S	Third	woman	False	NaN	Southampton	no	False	female	청년	A2	0
889	1	1	male	26.000000	0	0	30.00	C	First	man	True	C	Cherbourg	yes	True	male	청년	A2	1
890	0	3	male	32.000000	0	0	7.75	Q	Third	man	True	NaN	Queenstown	no	True	male	중년	A3	1
In [35]:
titanic.pivot_table('gender', 'age_group')
Out[35]:
gender
age_group	
A1	0.594684
A2	0.680921
A3	0.667832
In [36]:
titanic.pivot_table(['gender','survived'], 'age_group')
Out[36]:
gender	survived
age_group		
A1	0.594684	0.411960
A2	0.680921	0.335526
A3	0.667832	0.405594
3. Mile Per Gallon
a. 배기량(displacement) 대비 마력(horsepower) 열(hp_per_cc)을 추가하시오.
In [ ]:

b. name으로부터 manufacturer(제조사)와 모델을 추출하여 새로운 열 manufacturer와 model을 추가하고, name 열은 삭제하시오.
In [ ]:

In [ ]:

In [ ]:

In [ ]:

c. 엔진의 실린더(cylinders) 갯수별 연비(mpg)의 평균을 구하시오.
In [ ]:

In [ ]:

d. 생산지(origin)별 배기량 대비 마력(hp_per_cc)의 평균을 구하시오.
In [ ]:

e. 모델이 5개 이상인 제조사에 대하여 연비(mpg)의 평균이 가장 좋은 제조사 Top 5를 구하시오.
In [ ]:
