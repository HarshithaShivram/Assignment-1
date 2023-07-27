#!/usr/bin/env python
# coding: utf-8

# # Q7)
# Calculate Mean, Median, Mode, Variance, Standard Deviation, Range &     comment about the values / draw inferences, for the given dataset
# -	For Points,Score,Weigh>
# Find Mean, Median, Mode, Variance, Standard Deviation, and Range and also Comment about the values/ Draw some inferences.
# Use Q7.csv file 
# 

# In[ ]:


# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import kurtosis
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")


# In[2]:


cars= pd.read_csv("C:\\Users\\SONY1\\Documents\\Assignment-1\\Q7.csv")
cars


# In[3]:


#MEAN
cars.mean()


# In[4]:


#MEDIAN
cars.median()


# In[5]:


#MODE
cars["Points"].mode()


# In[6]:


cars["Score"].mode()


# In[7]:


cars["Weigh"].mode()


# In[8]:


#VARIANCE
cars.var()


# In[9]:


cars.std()


# In[10]:


#RANGE
cars.describe()


# In[11]:


points_range= cars["Points"].max()-cars["Points"].min()
points_range


# In[12]:


scores_range= cars["Score"].max()-cars["Score"].min()
scores_range


# In[13]:


weigh_range= cars["Weigh"].max()-cars["Weigh"].min()
weigh_range


# In[14]:


ax=plt.subplots(figsize=(10,4))
plt.subplot(1,3,1)
plt.boxplot(cars.Points)
plt.title("Points")
plt.subplot(1,3,2)
plt.boxplot(cars.Score)
plt.title("Score")
plt.subplot(1,3,3)
plt.boxplot(cars.Weigh)
plt.title("Weigh")
plt.show()


# # Q8) 
# Calculate Expected Value for the problem below
# a)The weights (X) of patients at a clinic (in pounds), are
# 108, 110, 123, 134, 135, 145, 167, 187, 199
# Assume one of the patients is chosen at random. What is the Expected Value of the Weight of that patient?
# 

# In[15]:


a = [108, 110, 123, 134, 135, 145, 167, 187, 199]
a


# In[16]:


sum(a)/len(a)


# # Q9(a)
# Calculate Skewness, Kurtosis & draw inferences on the following data
# Cars speed and distance 
# Use Q9_a.csv
# 

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
from scipy.stats import kurtosis
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")


# In[18]:


cars = pd.read_csv("C:\\Users\\SONY1\\Documents\\Assignment-1\\Q9_a.csv")
cars


# In[19]:


#SKEWNESS
cars.skew()


# In[20]:


#KURTOSIS
cars.kurt()


# In[21]:


sns.displot(data=cars["speed"],kind="kde")
sns.displot(data=cars["dist"],kind="kde")


# In[22]:


sns.distplot(cars['speed'],hist=False,color='blue')
sns.distplot(cars['dist'],hist=False,color='red')


# In[23]:


ax=plt.subplots(figsize=(12,4))
plt.subplot(1,4,1)
plt.boxplot(cars.speed)
plt.title("Speed")
plt.subplot(1,4,2)
plt.boxplot(cars.dist)
plt.title("Dist")
plt.show()


# # Q9(b)
# SP and Weight(WT) Use Q9_b.csv

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import kurtosis
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")


# In[25]:


car= pd.read_csv("C:\\Users\\SONY1\\Documents\\Assignment-1\\Q9_b.csv")
car


# In[26]:


#SKEWNESS
car.skew()


# In[27]:


#KURTOSIS 
car.kurtosis()


# In[28]:


sns.displot(data= car["SP"], kind = "kde")
sns.displot(data= car["WT"], kind = "kde")


# In[29]:


sns.distplot(car['SP'],hist=False,color='blue')
sns.distplot(car['WT'],hist=False,color='red')


# In[30]:


ax = plt.subplots(figsize = (12,4))
plt.subplot(1,4,1)
plt.boxplot(car.SP)
plt.title("SP")
plt.subplot(1,4,2)
plt.boxplot(car.WT)
plt.title("WT")
plt.show()


# # Q11)
# Suppose we want to estimate the average weight of an adult male in Mexico. We draw a random sample of 2,000 men from a population of 3,000,000 men and weigh them. We find that the average person in our sample weighs 200 pounds, and the standard deviation of the sample is 30 pounds. Calculate 94%,98%,96% confidence interval?

# In[31]:


import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


# In[32]:


#Average weight of Adult in Mexico with 94% CI
stats.norm.interval(0.94,200,30/(2000**0.5))


# In[33]:


#Average weight of Adult in Mexico with 98% CI 
stats.norm.interval(0.98,200,30/(2000**0.5))


# In[34]:


#Average weight of Adult in Mexico with 96% CI 
stats.norm.interval(0.96,200,30/(2000**0.5))


# # Q12)
# Below are the scores obtained by a student in tests 
# 34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56
# 1)	Find mean,median,variance,standard deviation.
# 2)	What can we say about the student marks? 
# 

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[36]:


scores = pd.Series([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])
scores


# In[37]:


scores.mean()


# In[38]:


scores.median()


# In[39]:


scores.var()


# In[40]:


scores.std()


# In[41]:


plt.boxplot(scores)
plt.grid()
plt.show()


# In[42]:


plt.hist(scores, color='b')
plt.grid()
plt.show()


# In[43]:


scores.skew()


# In[44]:


scores.kurt()

Mean is greater than median, so the data is slightly skewed towards right
# # Q 20)
# Calculate probability from the given dataset for the below cases
# 
# Data _set: Cars.csv
# Calculate the probability of MPG  ofCars for the below cases.
# MPG<- Cars$MPG
# 
# a.	P(MPG>38)
# b.	P(MPG<40)
# c.  P(20<MPG<50)
# 

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


# In[46]:


c = pd.read_csv("C:\\Users\\SONY1\\Downloads\\Cars.csv")
c


# In[47]:


c.describe()


# In[48]:


#P(MPG>38)
Prob_MPG_greater_than_38 = np.round(1-stats.norm.cdf(38, loc=c.MPG.mean(), scale= c.MPG.std()),3)
print('P(MPG>38)=',Prob_MPG_greater_than_38)


# In[49]:


#P(MPG<40)
Prob_MPG_less_than_40 = np.round(stats.norm.cdf(40, loc = c.MPG.mean(), scale = c.MPG.std()),3)
print('P(MPG>20)=',Prob_MPG_less_than_40)


# In[50]:


#P(MPG>20)
Prob_MPG_greater_than_20 = np.round(1-stats.norm.cdf(20, loc = c.MPG.mean(), scale = c.MPG.std()),3)
print('P(MPG>20)=',(Prob_MPG_greater_than_20))


# In[51]:


#P(MPG<50)
Prob_MPG_less_than_50 = np.round(stats.norm.cdf(50, loc=c.MPG.mean(), scale=c.MPG.std()),3)
print('P(MPG<50)=',(Prob_MPG_less_than_50))


# In[52]:


#P(20<MPG<50)
Prob_MPG_greaterthan20_and_lessthan50= (Prob_MPG_less_than_50) - (Prob_MPG_greater_than_20)
print('P(20<MPG<50)=',(Prob_MPG_greaterthan20_and_lessthan50))


# # Q 21(a) 
# Check whether the data follows normal distribution
# a)Check whether the MPG of Cars follows Normal Distribution 
#     Dataset: Cars.csv
# 

# In[58]:


carss = pd.read_csv("C:\\Users\\SONY1\\Downloads\\Cars.csv")
carss


# In[59]:


plt.hist(carss["MPG"], bins = 20, edgecolor=  'black')
plt.show()


# In[60]:


plt.boxplot(x= 'MPG', data = carss)
plt.show()


# In[62]:


stats.probplot(carss['MPG'], dist="norm", plot=plt)
plt.xlabel('MPG', color ='red')
plt.savefig('MPG of carss.png')
plt.show()


# In[64]:


sns.distplot(carss['MPG'], kde=True, bins =8)
plt.show()


# # 21(b)
# Check Whether the Adipose Tissue (AT) and Waist Circumference(Waist)  from wc-at data set  follows Normal Distribution 
#        Dataset: wc-at.csv
# 

# In[67]:


wc_at = pd.read_csv("C:\\Users\\SONY1\\Documents\\Assignment-1\\wc-at.csv")
wc_at


# In[69]:


plt.hist(wc_at['Waist'], edgecolor = 'black')
plt.show()


# In[72]:


sns.distplot(wc_at['Waist'],bins=10,kde = True)
plt.show()


# In[73]:


stats.probplot(wc_at['Waist'], dist = 'norm', plot = plt)
plt.xlabel('Waist', color= 'red')
plt.savefig('Waist.png')
plt.show()


# In[74]:


plt.hist(wc_at['AT'], edgecolor= 'black')
plt.show()


# In[75]:


sns.distplot(wc_at['AT'], bins = 8, kde=True)
plt.show()


# In[79]:


stats.probplot(wc_at['AT'], dist = 'norm', plot = plt)
plt.xlabel('AT')
plt.savefig('AT.png')
plt.show()


# # Q 22) 
# Calculate the Z scoresof  90% confidence interval,94% confidence interval, 60% confidence interval 

# In[81]:


from scipy import stats 
from scipy.stats import norm


# In[82]:


#Z-score of 90% confidence interval
stats.norm.ppf(0.95)


# In[84]:


#Z-score of 94% confidence interval
stats.norm.ppf(0.97)


# In[85]:


#Z-score of 60confidence interval
stats.norm.ppf(0.80)


# # Q 23)
# Calculate the t-scores of 95% confidence interval, 96% confidence interval, 99% confidence interval for sample size of 25

# In[90]:


from scipy import stats
from scipy.stats import norm


# In[91]:


#t-scores of 95% confidence interval for sample size of 25
#df = n-1 = 24
stats.t.ppf(0.975,24)  


# In[92]:


#t-scores of 96% confidence interval for sample size of 25
#df = n-1 = 24
stats.t.ppf(0.98,24)


# In[93]:


#t-scores of 96% confidence interval for sample size of 25
#df = n-1 = 24
stats.t.ppf(0.995,24)


# # Q 24)
# A Government company claims that an average light bulb lasts 270 days. A researcher randomly selects 18 bulbs for testing. The sampled bulbs last an average of 260 days, with a standard deviation of 90 days. If the CEO's claim were true, what is the probability that 18 randomly selected bulbs would have an average life of no more than 260 days
# Hint: 
# rcodept(tscore,df)  
#  df  degrees of freedom
# 

# In[95]:


from scipy import stats
from scipy.stats import norm
#Assume Null Hypothesis is: Ho = Avg life of Bulb >= 260 days
#Alternate Hypothesis is: Ha = Avg life of Bulb < 260 days
#Population mean = 270 days
#Sample mean = 260 days
#Sample Std Dev = 90days
#Sample(n) = 18 bulbs
#df = n - 1 = 18-1 = 17


# In[96]:


#find t-scores at x=260; t=(s_mean-P_mean)/(s_SD/sqrt(n))
t=(260-270)/(90/18**0.5)
t


# In[97]:


# Find P(X>=260) for null hypothesis
# p_value=1-stats.t.cdf(abs(t_scores),df=n-1)... Using cdf function
p_value=1-stats.t.cdf(abs(-0.4714),df=17)
p_value


# In[98]:


#  OR p_value=stats.t.sf(abs(t_score),df=n-1)... Using sf function
p_value=stats.t.sf(abs(-0.4714),df=17)
p_value

Probability that 18 randomly selected bulbs would have an average life of no more than 260 days is 32.17%
Assuming significance value α = 0.05 (Standard Value)(If p_value < α ; Reject HO and accept HA or vice-versa)
Thus, as p-value > α ; Accept HO i.e.
The CEO claims are false and the avg life of bulb > 260 days
# In[99]:


1 -(1 - 0.005)**5


# In[ ]:




