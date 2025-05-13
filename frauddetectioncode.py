import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('fraudTrain.csv', index_col=0)
df_test = pd.read_csv('fraudTest.csv',index_col=0)

df_train.head()

df_train.columns

row,col=df_train.shape
print(f'The dataset has {row} rows and {col} columns.')

print(df_train.describe())

df_train.info()

fraud_percentage=len(df_train[df_train['is_fraud']==1])/len(df_train)*100
non_fraud_percentage=100-fraud_percentage
print(f'The percentage of fraud is {fraud_percentage} % and the percentage of non-fraud is {non_fraud_percentage} %')
print('This shows how imbalanced is the dataset')

df_train.isna().sum()

len(df_train)==len(df_train.drop_duplicates())

df_train.head()

df_train['cc_num'].value_counts()

df_train[df_train['is_fraud']==1].cc_num.value_counts()

df_train.drop(columns=['cc_num'],inplace=True)
df_test.drop(columns=['cc_num'],inplace=True)

df_train.merchant.value_counts()

df_train[df_train['is_fraud']==1].merchant.value_counts().head(50)

df_train.drop(columns=['merchant'],inplace=True)
df_test.drop(columns=['merchant'],inplace=True)

df_train.category.value_counts()

df_train[df_train['is_fraud']==1].category.value_counts()

(1743+1713+915+843+618)/(df_train[df_train['is_fraud']==1].category.value_counts().sum())*100

df_train.groupby('category')['amt'].sum().sort_values(ascending=False)

(14460822.38+9307993.61+8625149.68+8351732.29+7173928.11)/(df_train.groupby('category')['amt'].sum().sum())*100

def convert_to_other(instance):
    if(instance in ['grocery_pos', 'shopping_net', 'misc_net','shopping_pos', 'gas_transport']):
        return instance
    else :
        return 'others'
df_train['category']=df_train['category'].apply(convert_to_other)
df_test['category']=df_test['category'].apply(convert_to_other)
df_train.category.value_counts()

df_train.head()

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse_output=False)
# drop='first' removes one column to avoid redundancy which also remove multicollinearity
# if we don't remove the input feature will have depencies. sum of values in the col will be 1 which can effect the performance

encoded_array = encoder.fit_transform(df_train[['category']])

encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['category']))

df_train = pd.concat([df_train.drop(columns=['category']), encoded_df], axis=1)


encoded_array = encoder.transform(df_test[['category']])

encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['category']))

df_test = pd.concat([df_test.drop(columns=['category']), encoded_df], axis=1)

df_train.drop(columns=['first','last'],inplace=True)
df_test.drop(columns=['first','last'],inplace=True)

def M_1_F_0(instance):
    if(instance=='M'):
        return 1;
    return 0;


df_train['gender']=df_train['gender'].apply(M_1_F_0)
df_test['gender']=df_test['gender'].apply(M_1_F_0)

df_train.street.value_counts()

df_train[df_train.is_fraud==1].city.value_counts()

df_train['zip'].value_counts()

df_train.drop(columns=['street','city','state','zip','dob','trans_num','city_pop'],inplace=True)
df_test.drop(columns=['street','city','state','zip','dob','trans_num','city_pop'],inplace=True)
df_train.head()

df_train.drop(columns=['unix_time'],inplace=True)
df_test.drop(columns=['unix_time'],inplace=True)

df_train.head()

df_train.job.value_counts()

df_train[df_train.is_fraud==1].job.value_counts()

(62+56+53+51+50)/(len(df_train[df_train.is_fraud==1]))*100

df_train.drop(columns=['job'],inplace=True)
df_test.drop(columns=['job'],inplace=True)
df_train.head()

ax=sns.histplot(x='amt',data=df_train[df_train.amt<=1000],hue='is_fraud',stat='percent',multiple='dodge',common_norm=False,bins=25)
ax.set_ylabel('Percentage in Each Type')
ax.set_xlabel('Transaction Amount in USD')
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])

df_train[(df_train.amt>=200)& (df_train.is_fraud==1)].amt.sum()/df_train[df_train.is_fraud==1].amt.sum()*100

def is_greater_than_200(amt):
    if(amt>=200):
        return 1;
    return 0;
df_train['is_amt_greater_than_200']=df_train.amt.apply(is_greater_than_200)
df_test['is_amt_greater_than_200']=df_test.amt.apply(is_greater_than_200)

split_sum = df_train[df_train.is_fraud==1].groupby('is_amt_greater_than_200')['amt'].sum()

plt.bar(split_sum.index.astype(str), split_sum.values, color=['blue', 'orange'])

df_train['hour']=pd.to_datetime(df_train['trans_date_trans_time']).dt.hour
df_test['hour']=pd.to_datetime(df_test['trans_date_trans_time']).dt.hour
ax=sns.histplot(data=df_train, x="hour", hue="is_fraud", common_norm=False,stat='percent',multiple='dodge')
ax.set_ylabel('Percentage')
ax.set_xlabel('Time (Hour) in a Day')
plt.xticks(np.arange(0,24,1))
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])

df_train[(df_train.is_fraud == 1) & ((df_train.hour <= 3) | (df_train.hour >= 22))]['amt'].sum()/(df_train[df_train.is_fraud==1].amt.sum())*100

def time_category(hour):
    if hour>=22 or hour<=3 :
        return "night";
    else :
        return "day"

df_train['time_category']= df_train['hour'].apply(time_category)
df_test['time_category']= df_test['hour'].apply(time_category)

df_train.time_category.value_counts()

df_train.info()

encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_array = encoder.fit_transform(df_train[['time_category']])

encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['time_category']))

df_train = pd.concat([df_train.drop(columns=['time_category']), encoded_df], axis=1)

encoded_array = encoder.transform(df_test[['time_category']])

encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['time_category']))

df_test = pd.concat([df_test.drop(columns=['time_category']), encoded_df], axis=1)

df_train.info()

df_train.head()

df_train['day']=pd.to_datetime(df_train['trans_date_trans_time']).dt.dayofweek
df_test['day']=pd.to_datetime(df_test['trans_date_trans_time']).dt.dayofweek
ax=sns.histplot(data=df_train, x="day", hue="is_fraud", common_norm=False,stat='percent',multiple='dodge')
#ax.set_xticklabels(['',"Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
plt.xticks(np.arange(7), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]) #set the xtick labels
ax.set_ylabel('Percentage')
ax.set_xlabel('Day of Week')
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])

df_train[(df_train.is_fraud == 1) & ((df_train.day >= 5)|(df_train.day==0))]['amt'].sum() / df_train[df_train.is_fraud == 1]['amt'].sum()*100

def category_day(day):
    if(day==0 or day>=5):
        return "cat1";
    else:
        return "cat2"
df_train['category_day']=df_train['day'].apply(category_day)
df_test['category_day']=df_test['day'].apply(category_day)

encoded_array = encoder.fit_transform(df_train[['category_day']])

encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['category_day']))

df_train = pd.concat([df_train.drop(columns=['category_day']), encoded_df], axis=1)

encoded_array = encoder.transform(df_test[['category_day']])

encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['category_day']))

df_test = pd.concat([df_test.drop(columns=['category_day']), encoded_df], axis=1)

df_test['category_day']=df_test['category_day_cat2']
df_test.drop(columns=['category_day_cat2'],inplace=True)

df_train['category_day']=df_train['category_day_cat2']
df_train.drop(columns=['category_day_cat2'],inplace=True)

df_train.category_day.value_counts()

df_fraud = df_train[df_train['is_fraud'] == 1]  # Keep only fraud transactions

ax = sns.histplot(data=df_fraud, x="day", common_norm=False, stat='percent', multiple='dodge')

ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_ylabel('Percentage')
ax.set_xlabel('Day of Week')
plt.title('Fraud Transactions by Day of the Week')
plt.show()

df_train['month']=pd.to_datetime(df_train['trans_date_trans_time']).dt.month
df_test['month']=pd.to_datetime(df_test['trans_date_trans_time']).dt.month
ax=sns.histplot(data=df_train, x="month", hue="is_fraud", common_norm=False,stat='percent',multiple='dodge')
ax.set_ylabel('Percentage')
ax.set_xlabel('Month')
plt.xticks(np.arange(1,13,1))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul",'Aug','Sep','Oct','Nov','Dec'])
plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])

df_fraud = df_train[df_train['is_fraud'] == 1]  # Keep only fraud transactions

ax = sns.histplot(data=df_fraud, x="month", common_norm=False, stat='percent', multiple='dodge')

ax.set_ylabel('Percentage')
ax.set_xlabel('Month')

plt.xticks(np.arange(1, 13, 1), ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

plt.title('Fraud Transactions by Month')
plt.show()

df_train.head()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

X_train = df_train.drop(columns=['trans_date_trans_time','day','hour','is_fraud','month']).copy()
y_train = df_train['is_fraud'].copy()

X_test = df_test.drop(columns=['trans_date_trans_time','day','hour','is_fraud','month']).copy()
y_test = df_test['is_fraud'].copy()

#Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

class FraudDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = FraudDataset(X_train_tensor, y_train_tensor)
test_dataset = FraudDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, pin_memory=True)

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryClassifier(X_train_tensor.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

#  Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        predicted = torch.sigmoid(outputs)
        predicted = (predicted > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")


from sklearn.metrics import precision_score, recall_score, f1_score

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        outputs = model(features)
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Convert predictions to binary labels
binary_preds = [1 if p > 0.5 else 0 for p in all_preds]

# Calculate metrics
precision = precision_score(all_labels, binary_preds)
recall = recall_score(all_labels, binary_preds)
f1 = f1_score(all_labels, binary_preds)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


amt_saved = df_test[(binary_preds==y_test) & (df_test.is_fraud==1)].amt.sum()

amt_saved

fraud_amt = df_test[df_test.is_fraud==1].amt.sum()

fraud_amt

saved_percentage=amt_saved/fraud_amt*100

print(f'Using this model saves {saved_percentage} % of fraud money')
