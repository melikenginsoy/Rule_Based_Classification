#######################################################################
# Potential Customer Profit Calculation with Rule-Based Classification
#######################################################################

###################
# Business Problem
###################
"""
A game company wants to create new customer definitions based on level by using
some features of its customers, create segments according to these new customer definitions (persona)
and estimate how much the new customers can earn according to these segments.

For Example:
It is desired to determine how much a 25-year-old male user from Turkey,
who is an IOS user, can earn on average.
"""

#####################
# About the Data Set
#####################

"""
The Persona.csv dataset contains the prices of the products sold by an international game company
and some demographic information of the users who buy these products.The data set consists of records 
created in each sales transaction. This means that the table is not deduplicated. In other words, 
a user with certain demographic characteristics may have made more than one purchase.
"""

###################
# Features
###################
"""
PRICE   – Customer spend amount
SOURCE  – The type of device the customer is connecting to
SEX     – Customer's gender
COUNTRY – Customer's country
AGE     – Customer's age
"""

# ------------------------------------------------------------------------------------------
import pandas as pd
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data = pd.read_csv("persona.csv")
df = data.copy()
df.head()

df.shape
# (5000, 5)

df.info()
"""
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   PRICE    5000 non-null   int64 
 1   SOURCE   5000 non-null   object
 2   SEX      5000 non-null   object
 3   COUNTRY  5000 non-null   object
 4   AGE      5000 non-null   int64 

"""

df.describe([0.1, 0.5, 0.75, 0.9, 0.95, 0.99]).T

"""
        count     mean        std   min   10%   50%   75%   90%   95%   99%   max
PRICE  5000.0  34.1320  12.464897   9.0  19.0  39.0  39.0  49.0  49.0  59.0  59.0
AGE    5000.0  23.5814   8.995908  15.0  15.0  21.0  27.0  36.0  43.0  53.0  66.0
"""
# How many unique SOURCE are there? What are their frequencies?
df["SOURCE"].nunique()
# 2

df["SOURCE"].value_counts()
"""
android    2974
ios        2026
"""

# How many sales were made from which PRICE?

df["PRICE"].value_counts()

"""
29    1305
39    1260
49    1031
19     992
59     212
9      200
"""

# How many sales from which country?

df["COUNTRY"].value_counts()

"""
usa    2065
bra    1496
deu     455
tur     451
fra     303
can     230
"""

# What are the PRICE averages in the COUNTRY-SOURCE breakdown?

df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

# What are the average earnings in breakdown of COUNTRY, SOURCE, SEX, AGE?

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE":"mean"})

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE":"mean"}).sort_values(by="PRICE", ascending=False)
agg_df
agg_df.reset_index(inplace=True)
agg_df.head()

"""
  COUNTRY   SOURCE     SEX  AGE  PRICE
0     bra  android    male   46   59.0
1     usa  android    male   36   59.0
2     fra  android  female   24   59.0
3     usa      ios    male   32   54.0
4     deu  android  female   36   49.0
"""


# Translating the age variable according to the categorical variable and adding it to agg_df.

bins=[0, 18, 23, 30, 40, 70]
labels=["0_18", "19_23", "24_30", "31_40", "41_70"]
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"],bins, labels=labels)
agg_df.head()

"""
  COUNTRY   SOURCE     SEX  AGE  PRICE AGE_CAT
0     bra  android    male   46   59.0   41_70
1     usa  android    male   36   59.0   31_40
2     fra  android  female   24   59.0   24_30
3     usa      ios    male   32   54.0   31_40
4     deu  android  female   36   49.0   31_40
"""

# Identifying new level-based customers (persona) and Deduplication of customers_level_based values

agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" +
                                   row[2].upper() + "_" + row[5].upper()
                                   for row in agg_df.values]
agg_df = agg_df.groupby(["customers_level_based"]).agg({"PRICE": "mean"}).sort_values(by="PRICE", ascending=False)

agg_df.head()

agg_df.reset_index(inplace=True)
agg_df.head()

"""
      customers_level_based      PRICE
0  FRA_ANDROID_FEMALE_24_30  45.428571
1        TUR_IOS_MALE_24_30  45.000000
2        TUR_IOS_MALE_31_40  42.333333
3  TUR_ANDROID_FEMALE_31_40  41.833333
4    CAN_ANDROID_MALE_19_23  40.111111
"""
# Segmenting personas

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.groupby(["SEGMENT"]).agg({"PRICE":["mean", "max", "sum"]})

"""
             PRICE                        
              mean        max          sum
SEGMENT                                   
D        29.206780  32.333333   817.789833
C        33.509674  34.077340   904.761209
B        34.999645  36.000000   944.990411
A        38.691234  45.428571  1044.663328
"""


# Classifying new customers and estimating how much revenue they can generate.
# • 33-year-old Turkish woman using ANDROID belongs to which segment?  How much income is expected to earn on average?

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"]== new_user]

"""
      customers_level_based      PRICE SEGMENT
3  TUR_ANDROID_FEMALE_31_40  41.833333       A
"""

# • A 35-year-old French woman using IOS belongs to which segment?  How much income is expected to earn on average?

new_user_1 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"]== new_user_1]

"""
   customers_level_based      PRICE SEGMENT
78  FRA_IOS_FEMALE_31_40  32.818182       C
"""