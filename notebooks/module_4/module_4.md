# Imports


```python
from dotenv import load_dotenv
import os
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from joblib import dump, load
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
local_path = "/home/lucas/zrive-ds/data/module_4/feature_frame.csv"
```

# Data Obtention


```python
'''
load_dotenv()
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

s3 = session.client("s3")

bucket_name = "zrive-ds-data"
path = "groceries/box_builder_dataset/feature_frame.csv"

try:
    s3.download_file(bucket_name, path, local_path)
    print(f"File downloaded succesfully.")
except Exception as e:
    print(f"File not found : {e}")
'''
```




    '\nload_dotenv()\nsession = boto3.Session(\n    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),\n    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),\n)\n\ns3 = session.client("s3")\n\nbucket_name = "zrive-ds-data"\npath = "groceries/box_builder_dataset/feature_frame.csv"\n\ntry:\n    s3.download_file(bucket_name, path, local_path)\n    print(f"File downloaded succesfully.")\nexcept Exception as e:\n    print(f"File not found : {e}")\n'



Lists with numeric and categorical columns are defined.


```python
numeric_cols = ['user_order_seq',
                'ordered_before',
                'abandoned_before',
                'active_snoozed',
                'set_as_regular',
                'normalised_price',
                'discount_pct',
                'global_popularity',
                'count_adults',
                'count_children',
                'count_babies',
                'count_pets',
                'people_ex_baby',
                'days_since_purchase_variant_id',
                'avg_days_to_buy_variant_id',
                'std_days_to_buy_variant_id',
                'days_since_purchase_product_type',
                'avg_days_to_buy_product_type',
                'std_days_to_buy_product_type'
                ]
categorical_cols = ['vendor',
                    'order_date',
                    'created_at',
                    'product_type']
```

Since it is a constraint, orders with 5 products or more are selected from the complete dataset.


```python
dataset = pd.read_csv(local_path)

orders = dataset[dataset['outcome'] == 1]

order_sizes = orders.groupby('order_id').size()

large_orders = order_sizes[order_sizes >= 5].index

selected_orders = dataset[dataset['order_id'].isin(large_orders)]

selected_orders.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808434524292</td>
      <td>3479090790532</td>
      <td>2020-10-06 10:50:23</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>6</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808548917380</td>
      <td>3476645445764</td>
      <td>2020-10-06 14:23:08</td>
      <td>2020-10-06 00:00:00</td>
      <td>5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>7</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808549900420</td>
      <td>3437017956484</td>
      <td>2020-10-06 14:24:26</td>
      <td>2020-10-06 00:00:00</td>
      <td>13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>9</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808728715396</td>
      <td>3421126885508</td>
      <td>2020-10-06 19:36:06</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>10</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808769085572</td>
      <td>3442602868868</td>
      <td>2020-10-06 20:45:38</td>
      <td>2020-10-06 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>11</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808786714756</td>
      <td>3486509793412</td>
      <td>2020-10-06 21:19:13</td>
      <td>2020-10-06 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>16</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2809127272580</td>
      <td>3771731083396</td>
      <td>2020-10-07 11:48:56</td>
      <td>2020-10-07 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>17</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2809189335172</td>
      <td>3223115595908</td>
      <td>2020-10-07 13:54:54</td>
      <td>2020-10-07 00:00:00</td>
      <td>9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>18</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2809486573700</td>
      <td>3418176061572</td>
      <td>2020-10-07 16:08:54</td>
      <td>2020-10-07 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>19</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2809696288900</td>
      <td>3510998990980</td>
      <td>2020-10-07 17:16:02</td>
      <td>2020-10-07 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>20</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2809857507460</td>
      <td>3395660513412</td>
      <td>2020-10-07 18:38:56</td>
      <td>2020-10-07 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>21</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2810124632196</td>
      <td>3766351462532</td>
      <td>2020-10-07 20:35:31</td>
      <td>2020-10-07 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>22</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2810929840260</td>
      <td>3489672626308</td>
      <td>2020-10-08 06:39:15</td>
      <td>2020-10-08 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>23</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2811003142276</td>
      <td>3432247033988</td>
      <td>2020-10-08 07:47:01</td>
      <td>2020-10-08 00:00:00</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>24</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2811116486788</td>
      <td>3463900528772</td>
      <td>2020-10-08 09:31:35</td>
      <td>2020-10-08 00:00:00</td>
      <td>5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>25</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2811176255620</td>
      <td>3315720519812</td>
      <td>2020-10-08 10:25:23</td>
      <td>2020-10-08 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 27 columns</p>
</div>



Variables weekday and time_of_day are created as it is considered they may add valuable information to the model.


```python
selected_orders['created_at'] = pd.to_datetime(selected_orders['created_at'])
selected_orders['weekday'] = selected_orders['created_at'].dt.dayofweek
categorical_cols.append('weekday')
```

    /tmp/ipykernel_1559/2781777034.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      selected_orders['created_at'] = pd.to_datetime(selected_orders['created_at'])
    /tmp/ipykernel_1559/2781777034.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      selected_orders['weekday'] = selected_orders['created_at'].dt.dayofweek



```python
selected_orders['hour'] = selected_orders['created_at'].dt.hour

def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 22:
        return 'evening'
    else:
        return 'night'

selected_orders['time_of_day'] = selected_orders['hour'].apply(get_time_of_day)
categorical_cols.append('time_of_day')
selected_orders.drop('hour', axis=1, inplace=True)

```

    /tmp/ipykernel_1559/278257896.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      selected_orders['hour'] = selected_orders['created_at'].dt.hour
    /tmp/ipykernel_1559/278257896.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      selected_orders['time_of_day'] = selected_orders['hour'].apply(get_time_of_day)
    /tmp/ipykernel_1559/278257896.py:15: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      selected_orders.drop('hour', axis=1, inplace=True)



```python
selected_orders.sort_values('order_id').head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
      <th>weekday</th>
      <th>time_of_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>481583</th>
      <td>33973246853252</td>
      <td>tinspackagedfoods</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>30.0</td>
      <td>30.234265</td>
      <td>30.0</td>
      <td>27.0</td>
      <td>23.827826</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2398555</th>
      <td>33667228663940</td>
      <td>wipescottonwool</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>41.5</td>
      <td>28.238356</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.826713</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>478137</th>
      <td>33863279214724</td>
      <td>bathshowergel</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>54.0</td>
      <td>35.319072</td>
      <td>30.0</td>
      <td>37.0</td>
      <td>30.506129</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2402001</th>
      <td>33826467152004</td>
      <td>superfoodssupplements</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>55.0</td>
      <td>34.085746</td>
      <td>30.0</td>
      <td>37.0</td>
      <td>27.032264</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>474691</th>
      <td>33803543347332</td>
      <td>foodstorage</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>14.0</td>
      <td>3.532165</td>
      <td>30.0</td>
      <td>37.5</td>
      <td>30.498356</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2405807</th>
      <td>33826458337412</td>
      <td>superfoodssupplements</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>32.0</td>
      <td>23.810595</td>
      <td>30.0</td>
      <td>37.0</td>
      <td>27.032264</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>471245</th>
      <td>34173018734724</td>
      <td>homebaking</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.5</td>
      <td>32.267135</td>
      <td>30.0</td>
      <td>28.5</td>
      <td>23.710730</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>467799</th>
      <td>33824368033924</td>
      <td>washingpowder</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>30.5</td>
      <td>28.724102</td>
      <td>30.0</td>
      <td>37.0</td>
      <td>29.593617</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2413965</th>
      <td>33667293085828</td>
      <td>juicesquash</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>27.0</td>
      <td>25.876853</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>464353</th>
      <td>33667228762244</td>
      <td>babytoiletries</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>43.0</td>
      <td>28.401115</td>
      <td>30.0</td>
      <td>36.0</td>
      <td>29.372186</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2417411</th>
      <td>33667260514436</td>
      <td>kidssnacks</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>27.0</td>
      <td>23.732701</td>
      <td>30.0</td>
      <td>26.5</td>
      <td>22.394709</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>460907</th>
      <td>34173020405892</td>
      <td>cookingingredientsoils</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>36.0</td>
      <td>29.704404</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>27.135844</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>457461</th>
      <td>34047322783876</td>
      <td>allpurposecleaner</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>14.0</td>
      <td>2.509980</td>
      <td>30.0</td>
      <td>36.0</td>
      <td>28.268085</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2425863</th>
      <td>34081589100676</td>
      <td>kidssnacks</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>31.0</td>
      <td>36.654377</td>
      <td>30.0</td>
      <td>26.5</td>
      <td>22.394709</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>454015</th>
      <td>33826423341188</td>
      <td>cereal</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>23.0</td>
      <td>16.256221</td>
      <td>30.0</td>
      <td>29.0</td>
      <td>26.476340</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2395109</th>
      <td>33667236331652</td>
      <td>catfood</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>37.0</td>
      <td>29.694523</td>
      <td>30.0</td>
      <td>27.0</td>
      <td>27.212938</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2429309</th>
      <td>34081590050948</td>
      <td>juicesquash</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>84.0</td>
      <td>39.142191</td>
      <td>30.0</td>
      <td>27.0</td>
      <td>25.876853</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>485029</th>
      <td>33667206938756</td>
      <td>dishwasherdetergent</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>43.0</td>
      <td>31.622258</td>
      <td>30.0</td>
      <td>32.0</td>
      <td>25.841947</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>495355</th>
      <td>33826413576324</td>
      <td>homebaking</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>37.0</td>
      <td>30.220274</td>
      <td>30.0</td>
      <td>28.5</td>
      <td>23.710730</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2350742</th>
      <td>33826439528580</td>
      <td>cookingingredientsoils</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>30.0</td>
      <td>20.858361</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>27.135844</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>528264</th>
      <td>33667232891012</td>
      <td>cookingingredientsoils</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>41.0</td>
      <td>31.369350</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>27.135844</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2354188</th>
      <td>33667184722052</td>
      <td>babytoiletries</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>18.5</td>
      <td>16.067566</td>
      <td>30.0</td>
      <td>36.0</td>
      <td>29.372186</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>524818</th>
      <td>34284953763972</td>
      <td>snacksconfectionery</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>35.0</td>
      <td>29.200482</td>
      <td>30.0</td>
      <td>27.0</td>
      <td>23.634873</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2357634</th>
      <td>33667235971204</td>
      <td>dogfood</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>23.0</td>
      <td>22.586869</td>
      <td>30.0</td>
      <td>24.0</td>
      <td>26.048133</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2361080</th>
      <td>33667182395524</td>
      <td>facialskincare</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>50.5</td>
      <td>20.513410</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>28.492200</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2364526</th>
      <td>34246817382532</td>
      <td>bathshowergel</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>51.0</td>
      <td>27.398297</td>
      <td>30.0</td>
      <td>37.0</td>
      <td>30.506129</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>514963</th>
      <td>33826427371652</td>
      <td>cookingingredientsoils</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>29.5</td>
      <td>31.646348</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>27.135844</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2367972</th>
      <td>33719434903684</td>
      <td>bodyskincare</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>25.0</td>
      <td>27.915746</td>
      <td>30.0</td>
      <td>39.5</td>
      <td>31.339570</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>2371418</th>
      <td>33719434182788</td>
      <td>bodyskincare</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>39.5</td>
      <td>31.339570</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 29 columns</p>
</div>



As we saw in eda module, 'days_since_purchase' varibles shows a strange pattern, so they will be eliminated.
Similar for count variables.


```python
to_drop_columns = ['days_since_purchase_variant_id',
                      'days_since_purchase_product_type',
                      'count_adults',
                      'count_children',
                      'count_babies',
                      'count_pets',
                      'people_ex_baby',
                      ]

for item in to_drop_columns:
    if item in numeric_cols:
        numeric_cols.remove(item)

selected_orders.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
      <th>weekday</th>
      <th>time_of_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>0</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>0</td>
      <td>evening</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>0</td>
      <td>evening</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>1</td>
      <td>morning</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808434524292</td>
      <td>3479090790532</td>
      <td>2020-10-06 10:50:23</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>1</td>
      <td>morning</td>
    </tr>
    <tr>
      <th>6</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808548917380</td>
      <td>3476645445764</td>
      <td>2020-10-06 14:23:08</td>
      <td>2020-10-06 00:00:00</td>
      <td>5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>1</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>7</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808549900420</td>
      <td>3437017956484</td>
      <td>2020-10-06 14:24:26</td>
      <td>2020-10-06 00:00:00</td>
      <td>13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>1</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>9</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808728715396</td>
      <td>3421126885508</td>
      <td>2020-10-06 19:36:06</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>1</td>
      <td>evening</td>
    </tr>
    <tr>
      <th>10</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808769085572</td>
      <td>3442602868868</td>
      <td>2020-10-06 20:45:38</td>
      <td>2020-10-06 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>1</td>
      <td>evening</td>
    </tr>
    <tr>
      <th>11</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808786714756</td>
      <td>3486509793412</td>
      <td>2020-10-06 21:19:13</td>
      <td>2020-10-06 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>1</td>
      <td>evening</td>
    </tr>
    <tr>
      <th>16</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2809127272580</td>
      <td>3771731083396</td>
      <td>2020-10-07 11:48:56</td>
      <td>2020-10-07 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>2</td>
      <td>morning</td>
    </tr>
    <tr>
      <th>17</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2809189335172</td>
      <td>3223115595908</td>
      <td>2020-10-07 13:54:54</td>
      <td>2020-10-07 00:00:00</td>
      <td>9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>2</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>18</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2809486573700</td>
      <td>3418176061572</td>
      <td>2020-10-07 16:08:54</td>
      <td>2020-10-07 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>2</td>
      <td>afternoon</td>
    </tr>
    <tr>
      <th>19</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2809696288900</td>
      <td>3510998990980</td>
      <td>2020-10-07 17:16:02</td>
      <td>2020-10-07 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>2</td>
      <td>evening</td>
    </tr>
    <tr>
      <th>20</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2809857507460</td>
      <td>3395660513412</td>
      <td>2020-10-07 18:38:56</td>
      <td>2020-10-07 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>2</td>
      <td>evening</td>
    </tr>
    <tr>
      <th>21</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2810124632196</td>
      <td>3766351462532</td>
      <td>2020-10-07 20:35:31</td>
      <td>2020-10-07 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>2</td>
      <td>evening</td>
    </tr>
    <tr>
      <th>22</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2810929840260</td>
      <td>3489672626308</td>
      <td>2020-10-08 06:39:15</td>
      <td>2020-10-08 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>3</td>
      <td>morning</td>
    </tr>
    <tr>
      <th>23</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2811003142276</td>
      <td>3432247033988</td>
      <td>2020-10-08 07:47:01</td>
      <td>2020-10-08 00:00:00</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>3</td>
      <td>morning</td>
    </tr>
    <tr>
      <th>24</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2811116486788</td>
      <td>3463900528772</td>
      <td>2020-10-08 09:31:35</td>
      <td>2020-10-08 00:00:00</td>
      <td>5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>3</td>
      <td>morning</td>
    </tr>
    <tr>
      <th>25</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2811176255620</td>
      <td>3315720519812</td>
      <td>2020-10-08 10:25:23</td>
      <td>2020-10-08 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>3</td>
      <td>morning</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 29 columns</p>
</div>



# Dataset Split

Train, validation and test split is done (70%, 10% and 20% of the original dataset). Time Series nature of data is considered as it has been divided in sequential splits. Possible order split has been taken in care


```python
unique_orders = selected_orders.groupby('order_id')['created_at'].min().sort_values()
train_size = int(len(unique_orders) * 0.7)
val_size = int(len(unique_orders) * 0.1)

train_orders = unique_orders.index[:train_size]
val_orders = unique_orders.index[train_size:train_size + val_size]
test_orders = unique_orders.index[train_size + val_size:]

X = selected_orders.drop('outcome', axis=1)
y = selected_orders['outcome']

X_train = X[X['order_id'].isin(train_orders)]
X_val = X[X['order_id'].isin(val_orders)]
X_test = X[X['order_id'].isin(test_orders)]

y_train = y[X_train.index]
y_val = y[X_val.index]
y_test = y[X_test.index]

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

```

    Training samples: 1446691
    Validation samples: 229825
    Testing samples: 487437


Frequency encoding is done in variables 'product_type', 'type_of_day' and 'vendor'.


```python
def frequency_encoding(
    column: str,
    column_list: list = categorical_cols
):
    freq_encoding = X_train[column].value_counts(normalize=True)

    X_train[column + '_freq'] = X_train[column].map(freq_encoding)

    X_val[column + '_freq'] = X_val[column].map(freq_encoding)

    X_test[column + '_freq'] = X_test[column].map(freq_encoding)

    column_list.append(column + '_freq')
    column_list.remove(column)
    return column_list

```


```python
categorical_cols = frequency_encoding('product_type')
categorical_cols = frequency_encoding('time_of_day')
categorical_cols = frequency_encoding('vendor')

```

    /tmp/ipykernel_1559/3438280891.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_train[column + '_freq'] = X_train[column].map(freq_encoding)
    /tmp/ipykernel_1559/3438280891.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_val[column + '_freq'] = X_val[column].map(freq_encoding)
    /tmp/ipykernel_1559/3438280891.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_test[column + '_freq'] = X_test[column].map(freq_encoding)
    /tmp/ipykernel_1559/3438280891.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_train[column + '_freq'] = X_train[column].map(freq_encoding)
    /tmp/ipykernel_1559/3438280891.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_val[column + '_freq'] = X_val[column].map(freq_encoding)
    /tmp/ipykernel_1559/3438280891.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_test[column + '_freq'] = X_test[column].map(freq_encoding)
    /tmp/ipykernel_1559/3438280891.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_train[column + '_freq'] = X_train[column].map(freq_encoding)
    /tmp/ipykernel_1559/3438280891.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_val[column + '_freq'] = X_val[column].map(freq_encoding)
    /tmp/ipykernel_1559/3438280891.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X_test[column + '_freq'] = X_test[column].map(freq_encoding)



```python
categorical_cols.remove('order_date')
categorical_cols.remove('created_at')
X_train[categorical_cols].head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weekday</th>
      <th>product_type_freq</th>
      <th>time_of_day_freq</th>
      <th>vendor_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.046227</td>
      <td>0.344312</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.046227</td>
      <td>0.298998</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.046227</td>
      <td>0.298998</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.046227</td>
      <td>0.272465</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0.046227</td>
      <td>0.272465</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0.046227</td>
      <td>0.344312</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0.046227</td>
      <td>0.344312</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0.046227</td>
      <td>0.298998</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0.046227</td>
      <td>0.298998</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0.046227</td>
      <td>0.298998</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2</td>
      <td>0.046227</td>
      <td>0.272465</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2</td>
      <td>0.046227</td>
      <td>0.344312</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>0.046227</td>
      <td>0.344312</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2</td>
      <td>0.046227</td>
      <td>0.298998</td>
      <td>0.015113</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>0.046227</td>
      <td>0.298998</td>
      <td>0.015113</td>
    </tr>
  </tbody>
</table>
</div>



# Definition of Functions

Different functions are defined to train models and evaluate them.


```python
def curves_computation(
    y_pred,
    y_val: pd.Series,
    ax1,
    ax2,
    name
):
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_val, y_pred)
    pr_auc = auc(recall, precision)

    ax1.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.5f})')
    
    ax2.plot(recall, precision, lw=2, label=f'{name} (AUC = {pr_auc:.5f})')
    
    return ax1, ax2
    
```


```python
def paint_graphics(models, cols:list, model_type: str = None):
    """
    Plot ROC and Precision-Recall curves for model evaluation and comparison.
    
    This function handles three different scenarios:
    1. GridSearchCV: Fits the grid search, plots top 5 models, and returns best estimator
    2. Dictionary of models with 'train + test': Trains models and evaluates on test set
    3. Dictionary of models with 'test': Evaluates pre-trained models on test set
    
    Parameters
    ----------
    models : GridSearchCV or dict
        Either a GridSearchCV object for hyperparameter tuning, or a dictionary 
        containing model instances with their names as keys.
    cols : list
        List of column names (features) to use for training and prediction.
    model_type : str, optional
        Specifies the model type and evaluation mode. Options:
        - 'decision_tree': For DecisionTreeClassifier GridSearchCV
        - 'gradient_boosting': For GradientBoostingClassifier GridSearchCV  
        - 'random_forest': For RandomForestClassifier GridSearchCV
        - 'xgboost': For XGBClassifier GridSearchCV
        - 'train + test': Train models on X_train and evaluate on X_test
        - 'test': Evaluate pre-trained models on X_test
        - None: Default behavior
    
    Returns
    -------
    sklearn estimator or None
        If models is a GridSearchCV object, returns the best estimator found.
        Otherwise, returns None.
    
    Notes
    -----
    - The function expects global variables: X_train, y_train, X_val, y_val, X_test, y_test
    - Plots are displayed using matplotlib with ROC curves on the left and 
      Precision-Recall curves on the right
    - For GridSearchCV, up to 5 top performing models are plotted
    - Random baseline (diagonal line) is included in ROC plot for reference
    
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot([0, 1], [0, 1], 'k--', label='Random baseline')
    

    if isinstance(models, GridSearchCV):
        models.fit(X_train[cols], y_train)
        results = pd.DataFrame(models.cv_results_)
        results = results.sort_values('mean_test_score', ascending=False)
        top_val_score = models.best_score_
        print(f'Top validation ROC_AUC: {top_val_score}')
        
        for i in range(5):
            if i < len(results):
                params = results.iloc[i]['params']
                if model_type == 'decission_tree':
                    model = DecisionTreeClassifier(**params)
                elif model_type == 'gradient_boosting':
                    model = GradientBoostingClassifier(**params,random_state=42)
                elif model_type == 'random_forest':
                    model = RandomForestClassifier(**params,random_state=42)
                elif model_type == 'xgboost':
                    model = XGBClassifier(**params,random_state=42)
                
                model.fit(X_train[cols], y_train)
                y_pred = model.predict_proba(X_val[cols])[:, 1]
                ax1, ax2 = curves_computation(y_pred, y_val, ax1, ax2, f'Top {i+1} {model_type} model')
    
    
    elif model_type == 'train + test':
        for name, model in models.items():    
            model.fit(X_train[cols], y_train)
            y_pred = model.predict_proba(X_test[cols])[:, 1]
            ax1, ax2 = curves_computation(y_pred, y_test, ax1, ax2,name)
    
    elif model_type == 'test':
        for name, model in models.items():
            y_pred = model.predict_proba(X_test[cols])[:, 1]
            ax1, ax2 = curves_computation(y_pred, y_test, ax1, ax2,name)
    
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves Comparison')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves Comparison')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    if isinstance(models, GridSearchCV):
        return models.best_estimator_
```

A function is defined to take a fitted model and evaluate its features.


```python
def plot_feature_importance(model, feature_names):
    
    if not hasattr(model, 'feature_importances_'):
        print("El modelo no tiene atributo 'feature_importances_'")
        return
    
    importances = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    
    plt.figure(figsize=(10, 8))
    
    bars = plt.barh(range(len(feature_importance_df)), 
                    feature_importance_df['importance'], 
                    color='skyblue', 
                    edgecolor='navy', 
                    alpha=0.7)
    
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('Importancia')
    plt.ylabel('Features')
    plt.title(f'Features más Importantes')
    plt.gca().invert_yaxis()
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + max(feature_importance_df['importance']) * 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'{width:.4f}',
                ha='left', va='center', fontsize=9)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
    return feature_importance_df
```

ROC curve and precission-recall curve are used as metrics of performance because we don´t know which will be threshold used to classificate and the class imbalance can distort metrics like accuracy. 

# Baseline Definition

First of all, a logistic regression model is trained with the hyperparameters selected in the previous model. The goal will be to improve the perfomance of this baseline model.


```python
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l1',C=1,random_state=42, solver='liblinear')
)
to_be_tested = {'LR':pipeline}
paint_graphics(to_be_tested, numeric_cols, 'train + test')
```


    
![png](module_4_files/module_4_30_0.png)
    


# Model Training and Evaluation

From now on, different non-linear models will be trained and evaluated.

## Decission Tree

Two decission trees are trained, one with numeric columns and other with numeric and categorical columns. The results from both models will be compared.


```python
dt = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [40, 60, 80],
    'criterion': ['log_loss', 'entropy']
}

tscv = TimeSeriesSplit(n_splits=3)

dt_grid = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    scoring='roc_auc',
    cv = tscv,
    n_jobs=-1,            
    verbose=1
)

best_decision_tree_num = paint_graphics(dt_grid, numeric_cols, 'decission_tree')
```

    Fitting 3 folds for each of 48 candidates, totalling 144 fits


    /home/lucas/zrive-ds/.venv/lib/python3.11/site-packages/numpy/ma/core.py:2820: RuntimeWarning: invalid value encountered in cast
      _data = np.array(data, dtype=dtype, copy=copy,


    Top validation ROC_AUC: 0.8168865018403612



    
![png](module_4_files/module_4_35_3.png)
    



```python
best_decision_tree_numcat = paint_graphics(dt_grid, numeric_cols+categorical_cols,'decission_tree')
```

    Fitting 3 folds for each of 48 candidates, totalling 144 fits


    /home/lucas/zrive-ds/.venv/lib/python3.11/site-packages/numpy/ma/core.py:2820: RuntimeWarning: invalid value encountered in cast
      _data = np.array(data, dtype=dtype, copy=copy,


    Top validation ROC_AUC: 0.8195636650232299



    
![png](module_4_files/module_4_36_3.png)
    



```python
print(best_decision_tree_num)
plot_feature_importance(best_decision_tree_num,numeric_cols)
```

    DecisionTreeClassifier(criterion='log_loss', max_depth=7, min_samples_leaf=40,
                           random_state=42)



    
![png](module_4_files/module_4_37_1.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>ordered_before</td>
      <td>0.424756</td>
    </tr>
    <tr>
      <th>7</th>
      <td>global_popularity</td>
      <td>0.417252</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abandoned_before</td>
      <td>0.124106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>set_as_regular</td>
      <td>0.015212</td>
    </tr>
    <tr>
      <th>10</th>
      <td>avg_days_to_buy_product_type</td>
      <td>0.006157</td>
    </tr>
    <tr>
      <th>9</th>
      <td>std_days_to_buy_variant_id</td>
      <td>0.004597</td>
    </tr>
    <tr>
      <th>8</th>
      <td>avg_days_to_buy_variant_id</td>
      <td>0.004055</td>
    </tr>
    <tr>
      <th>0</th>
      <td>user_order_seq</td>
      <td>0.002144</td>
    </tr>
    <tr>
      <th>5</th>
      <td>normalised_price</td>
      <td>0.000919</td>
    </tr>
    <tr>
      <th>6</th>
      <td>discount_pct</td>
      <td>0.000434</td>
    </tr>
    <tr>
      <th>11</th>
      <td>std_days_to_buy_product_type</td>
      <td>0.000369</td>
    </tr>
    <tr>
      <th>3</th>
      <td>active_snoozed</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(best_decision_tree_numcat)
plot_feature_importance(best_decision_tree_numcat,numeric_cols+categorical_cols)
```

    DecisionTreeClassifier(criterion='log_loss', max_depth=7, min_samples_leaf=60,
                           random_state=42)



    
![png](module_4_files/module_4_38_1.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>ordered_before</td>
      <td>0.424637</td>
    </tr>
    <tr>
      <th>7</th>
      <td>global_popularity</td>
      <td>0.416226</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abandoned_before</td>
      <td>0.124071</td>
    </tr>
    <tr>
      <th>4</th>
      <td>set_as_regular</td>
      <td>0.013351</td>
    </tr>
    <tr>
      <th>9</th>
      <td>std_days_to_buy_variant_id</td>
      <td>0.004274</td>
    </tr>
    <tr>
      <th>15</th>
      <td>vendor_freq</td>
      <td>0.003716</td>
    </tr>
    <tr>
      <th>8</th>
      <td>avg_days_to_buy_variant_id</td>
      <td>0.003715</td>
    </tr>
    <tr>
      <th>10</th>
      <td>avg_days_to_buy_product_type</td>
      <td>0.003575</td>
    </tr>
    <tr>
      <th>14</th>
      <td>time_of_day_freq</td>
      <td>0.001729</td>
    </tr>
    <tr>
      <th>0</th>
      <td>user_order_seq</td>
      <td>0.001388</td>
    </tr>
    <tr>
      <th>11</th>
      <td>std_days_to_buy_product_type</td>
      <td>0.001102</td>
    </tr>
    <tr>
      <th>5</th>
      <td>normalised_price</td>
      <td>0.000885</td>
    </tr>
    <tr>
      <th>13</th>
      <td>product_type_freq</td>
      <td>0.000698</td>
    </tr>
    <tr>
      <th>12</th>
      <td>weekday</td>
      <td>0.000492</td>
    </tr>
    <tr>
      <th>6</th>
      <td>discount_pct</td>
      <td>0.000140</td>
    </tr>
    <tr>
      <th>3</th>
      <td>active_snoozed</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = {'DT':DecisionTreeClassifier(criterion='log_loss', max_depth=7, min_samples_leaf=60,random_state=42)}
selected_cols = ['ordered_before','global_popularity','abandoned_before']
paint_graphics(model,selected_cols,'train + test')
```


    
![png](module_4_files/module_4_39_0.png)
    


Results from both models are not different and the importance of features does not change remarkably. The training with the three most important features seemed to show similar results. The model trained with numerical variables will be compared with other algorithms.


```python
best_decision_tree = best_decision_tree_num
```

## Gradient Boosting


```python
gb = GradientBoostingClassifier(random_state=42)

param_grid = {
    'learning_rate': [0.01, 0.1],        
    'n_estimators': [100, 200, 300] 
}

tscv = TimeSeriesSplit(n_splits=3)

gb_grid = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    cv=tscv,
    scoring='roc_auc',
    n_jobs=-1
)

best_gradient_boost = paint_graphics(gb_grid, numeric_cols, 'gradient_boosting')
```

    Top validation ROC_AUC: 0.8300506948176215



    
![png](module_4_files/module_4_43_1.png)
    



```python
print(best_gradient_boost)
plot_feature_importance(best_gradient_boost,numeric_cols)
```

    GradientBoostingClassifier(n_estimators=200, random_state=42)



    
![png](module_4_files/module_4_44_1.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>ordered_before</td>
      <td>0.467095</td>
    </tr>
    <tr>
      <th>7</th>
      <td>global_popularity</td>
      <td>0.269524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abandoned_before</td>
      <td>0.135815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>set_as_regular</td>
      <td>0.034646</td>
    </tr>
    <tr>
      <th>0</th>
      <td>user_order_seq</td>
      <td>0.022403</td>
    </tr>
    <tr>
      <th>8</th>
      <td>avg_days_to_buy_variant_id</td>
      <td>0.021063</td>
    </tr>
    <tr>
      <th>11</th>
      <td>std_days_to_buy_product_type</td>
      <td>0.011388</td>
    </tr>
    <tr>
      <th>10</th>
      <td>avg_days_to_buy_product_type</td>
      <td>0.010552</td>
    </tr>
    <tr>
      <th>5</th>
      <td>normalised_price</td>
      <td>0.010017</td>
    </tr>
    <tr>
      <th>9</th>
      <td>std_days_to_buy_variant_id</td>
      <td>0.007416</td>
    </tr>
    <tr>
      <th>3</th>
      <td>active_snoozed</td>
      <td>0.005595</td>
    </tr>
    <tr>
      <th>6</th>
      <td>discount_pct</td>
      <td>0.004486</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = {'GB':GradientBoostingClassifier(n_estimators=200,learning_rate=0.01)}
selected_cols = ['ordered_before','global_popularity','abandoned_before']
paint_graphics(model,selected_cols,'train + test')
```


    
![png](module_4_files/module_4_45_0.png)
    


Gradient Boosting shows a similar behaviour to decission tree and the importance of features is practically the same.

## Random Forest


```python
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20]
}

tscv = TimeSeriesSplit(n_splits=3)

rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,               
    scoring='roc_auc',  
    n_jobs=-1,           
    verbose=2
)

best_random_forest = paint_graphics(rf_grid, numeric_cols,'random_forest')
```

    Fitting 3 folds for each of 6 candidates, totalling 18 fits
    [CV] END .....................max_depth=10, n_estimators=100; total time=  28.2s
    [CV] END .....................max_depth=20, n_estimators=100; total time=  42.3s
    [CV] END .....................max_depth=10, n_estimators=200; total time=  52.7s
    [CV] END .....................max_depth=10, n_estimators=100; total time=  54.8s
    [CV] END .....................max_depth=10, n_estimators=300; total time= 1.3min
    [CV] END .....................max_depth=20, n_estimators=100; total time= 1.4min
    [CV] END .....................max_depth=10, n_estimators=100; total time= 1.6min
    [CV] END .....................max_depth=20, n_estimators=200; total time= 1.4min
    [CV] END .....................max_depth=10, n_estimators=200; total time= 1.9min
    [CV] END .....................max_depth=20, n_estimators=100; total time= 2.2min
    [CV] END .....................max_depth=10, n_estimators=300; total time= 2.6min
    [CV] END .....................max_depth=20, n_estimators=300; total time= 1.7min
    [CV] END .....................max_depth=10, n_estimators=200; total time= 2.7min
    [CV] END .....................max_depth=20, n_estimators=200; total time= 2.4min
    [CV] END .....................max_depth=10, n_estimators=300; total time= 3.5min
    [CV] END .....................max_depth=20, n_estimators=200; total time= 3.1min
    [CV] END .....................max_depth=20, n_estimators=300; total time= 2.7min
    [CV] END .....................max_depth=20, n_estimators=300; total time= 3.8min


    /home/lucas/zrive-ds/.venv/lib/python3.11/site-packages/numpy/ma/core.py:2820: RuntimeWarning: invalid value encountered in cast
      _data = np.array(data, dtype=dtype, copy=copy,


    Top validation ROC_AUC: 0.8102381333348155



    
![png](module_4_files/module_4_48_3.png)
    



```python
print(best_random_forest)
plot_feature_importance(best_random_forest,numeric_cols)
```

    RandomForestClassifier(max_depth=10, n_estimators=300, random_state=42)



    
![png](module_4_files/module_4_49_1.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>ordered_before</td>
      <td>0.305918</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abandoned_before</td>
      <td>0.281738</td>
    </tr>
    <tr>
      <th>7</th>
      <td>global_popularity</td>
      <td>0.183171</td>
    </tr>
    <tr>
      <th>4</th>
      <td>set_as_regular</td>
      <td>0.035733</td>
    </tr>
    <tr>
      <th>8</th>
      <td>avg_days_to_buy_variant_id</td>
      <td>0.029839</td>
    </tr>
    <tr>
      <th>11</th>
      <td>std_days_to_buy_product_type</td>
      <td>0.029731</td>
    </tr>
    <tr>
      <th>0</th>
      <td>user_order_seq</td>
      <td>0.028715</td>
    </tr>
    <tr>
      <th>9</th>
      <td>std_days_to_buy_variant_id</td>
      <td>0.027126</td>
    </tr>
    <tr>
      <th>5</th>
      <td>normalised_price</td>
      <td>0.024647</td>
    </tr>
    <tr>
      <th>10</th>
      <td>avg_days_to_buy_product_type</td>
      <td>0.023265</td>
    </tr>
    <tr>
      <th>6</th>
      <td>discount_pct</td>
      <td>0.019285</td>
    </tr>
    <tr>
      <th>3</th>
      <td>active_snoozed</td>
      <td>0.010832</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = {'RF': RandomForestClassifier(max_depth=10, n_estimators=300, random_state=42)}
selected_cols = ['ordered_before','global_popularity','abandoned_before']
paint_graphics(model, selected_cols, 'train + test')
```


    
![png](module_4_files/module_4_50_0.png)
    


## XGBoost


```python
xgb = XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    eval_metric='auc'
)

param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

tscv = TimeSeriesSplit(n_splits=3)

xgb_grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

best_xgboost = paint_graphics(xgb_grid, numeric_cols, 'xgboost')
```

    Fitting 3 folds for each of 8 candidates, totalling 24 fits
    Top validation ROC_AUC: 0.8291931221559651



    
![png](module_4_files/module_4_52_1.png)
    



```python
print(best_xgboost)
plot_feature_importance(best_xgboost, numeric_cols)
```

    XGBClassifier(base_score=None, booster=None, callbacks=None,
                  colsample_bylevel=None, colsample_bynode=None,
                  colsample_bytree=None, device=None, early_stopping_rounds=None,
                  enable_categorical=False, eval_metric='auc', feature_types=None,
                  feature_weights=None, gamma=None, grow_policy=None,
                  importance_type=None, interaction_constraints=None,
                  learning_rate=0.1, max_bin=None, max_cat_threshold=None,
                  max_cat_to_onehot=None, max_delta_step=None, max_depth=3,
                  max_leaves=None, min_child_weight=None, missing=nan,
                  monotone_constraints=None, multi_strategy=None, n_estimators=200,
                  n_jobs=None, num_parallel_tree=None, ...)



    
![png](module_4_files/module_4_53_1.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>abandoned_before</td>
      <td>0.549859</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ordered_before</td>
      <td>0.211341</td>
    </tr>
    <tr>
      <th>7</th>
      <td>global_popularity</td>
      <td>0.151746</td>
    </tr>
    <tr>
      <th>10</th>
      <td>avg_days_to_buy_product_type</td>
      <td>0.028967</td>
    </tr>
    <tr>
      <th>4</th>
      <td>set_as_regular</td>
      <td>0.024461</td>
    </tr>
    <tr>
      <th>3</th>
      <td>active_snoozed</td>
      <td>0.007740</td>
    </tr>
    <tr>
      <th>9</th>
      <td>std_days_to_buy_variant_id</td>
      <td>0.005452</td>
    </tr>
    <tr>
      <th>8</th>
      <td>avg_days_to_buy_variant_id</td>
      <td>0.004991</td>
    </tr>
    <tr>
      <th>0</th>
      <td>user_order_seq</td>
      <td>0.004686</td>
    </tr>
    <tr>
      <th>11</th>
      <td>std_days_to_buy_product_type</td>
      <td>0.004098</td>
    </tr>
    <tr>
      <th>5</th>
      <td>normalised_price</td>
      <td>0.003842</td>
    </tr>
    <tr>
      <th>6</th>
      <td>discount_pct</td>
      <td>0.002817</td>
    </tr>
  </tbody>
</table>
</div>



## Model Comparison

The best models for each algorithm are tested with test set and compared with each other and with the baseline to decide which of them will be used in production.


```python
baseline = pipeline.fit(X_train[numeric_cols], y_train)
trained_models = {
    'DT':best_decision_tree,
    'RF':best_random_forest,
    'GB':best_gradient_boost,
    'XGB':best_xgboost,
    'Baseline': baseline
}

paint_graphics(trained_models,numeric_cols,'test')
```


    
![png](module_4_files/module_4_56_0.png)
    


Threshold will be determined using the associated recall.


```python
probs = best_gradient_boost.predict_proba(X_test[numeric_cols])[:,1]
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    
    # Encontrar el índice del recall más cercano al deseado
idx = np.argmin(np.abs(recalls - 0.6))

# Si el índice es el último, usar el penúltimo threshold
if idx == len(thresholds):
    idx = len(thresholds) - 1

threshold = thresholds[idx]
actual_recall = recalls[idx]
precision = precisions[idx]

print(f'Threshold: {threshold}')
print(f'Recall: {actual_recall}')
```

    Threshold: 0.021798193308869247
    Recall: 0.600030969340353


#### Conclusions

- Given the results with test set, seems correct to select Gradient Boosting as the model used in production. However, any of the non linear models significantly improve the baseline.

- The optimum hyperparameters for gradient boosting are 200 estimators and learning rate = 0.01.

- The threshold used is determined taking in account what we think is an acceptable recall (around 0.6). This value results to be around 0.217

- For the selected threshold even the baseline model seems to behave similar to other models.
