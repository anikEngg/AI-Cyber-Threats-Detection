#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_curve, auc


# In[23]:


f1 = pd.read_csv("D:\\AI Cyber Project\\archive\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# f2 = pd.read_csv('/kaggle/input/network-intrusion-dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
# f3 = pd.read_csv('/kaggle/input/network-intrusion-dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv')
# f4 = pd.read_csv('/kaggle/input/network-intrusion-dataset/Monday-WorkingHours.pcap_ISCX.csv')
# f5 = pd.read_csv('/kaggle/input/network-intrusion-dataset/Monday-WorkingHours.pcap_ISCX.csv')
# f6 = pd.read_csv('/kaggle/input/network-intrusion-dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
# f7 = pd.read_csv('/kaggle/input/network-intrusion-dataset/Tuesday-WorkingHours.pcap_ISCX.csv')
# f8 = pd.read_csv('/kaggle/input/network-intrusion-dataset/Wednesday-workingHours.pcap_ISCX.csv')


# In[24]:


combine_df = pd.concat([f1], ignore_index=True)


# In[25]:


combine_df.head()


# In[26]:


combine_df.tail()


# In[27]:


combine_df.columns


# In[28]:


combine_df[' Label'].value_counts().sum


# In[29]:


encoder = LabelEncoder()
combine_df[' Label']= encoder.fit_transform(combine_df[' Label'])


# In[30]:


combine_df.head()


# In[31]:


df = combine_df.fillna(0)  # Replace NaN with 0
df


# In[32]:


# Check for NaNs
nan_mask = df.isna()
print("NaNs in DataFrame:\n", df[nan_mask].sum())

# Check for infinities
inf_mask = df.isin([np.inf, -np.inf])
print("Infs in DataFrame:\n", df[inf_mask].sum())


# In[33]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
df.fillna(0, inplace=True)  # Replace NaNs with 0


# In[34]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
df.dropna(inplace=True)  # Drop rows with NaNs

# If you want to drop columns with NaNs
# df.dropna(axis=1, inplace=True)


# In[35]:


df.isnull().sum()


# In[36]:


df=df.astype(int)
df


# In[37]:


X = df.drop(' Label',axis=1)
y = df[' Label']


# In[38]:


X, y


# In[39]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[40]:


X_scaled


# In[43]:


# Impute missing values (replace NaNs with the mean)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Determine the number of columns (features) in your DataFrame
num_columns = df.shape[1]

# Set an appropriate value for k (less than or equal to the number of columns)
k = min(10, num_columns)  # Adjust this as needed

# Initialize SelectKBest with the scoring function
k_best = SelectKBest(score_func=f_classif, k=k)

# Fit and transform the imputed data to select the top 10 features
X_new = k_best.fit_transform(X_imputed, y)


# In[44]:


# Get the boolean mask of selected features
selected_features_mask = k_best.get_support()
selected_features_mask 


# In[45]:


elected_feature_names = X.columns[selected_features_mask]
elected_feature_names


# In[46]:


new_columns=[' Destination Port', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packets/s', ' Min Packet Length',
       ' PSH Flag Count', ' URG Flag Count', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' min_seg_size_forward']


# In[47]:


df_new=X[new_columns]
df_new


# In[48]:


df_new['label']=df[' Label']
df_new['label']


# In[49]:


X1=df_new.iloc[:,:-1].values
y1=df_new.iloc[:,-1].values


# In[50]:


X1, y1


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=42)


# In[52]:


LR_model = LogisticRegression()

# Train the model
LR_model.fit(X_train, y_train)


# In[53]:


LR_y_pred = LR_model.predict(X_test)


# In[54]:


# For ROC-AUC, you need predicted probabilities, not just class labels
LR_y_prob = LR_model.decision_function(X_test)  # Use decision_function for SVM or predict_proba for other models


# In[55]:


LR_report = classification_report(y_test, LR_y_pred)
print("Classification Report:", LR_report)


# In[56]:


LR_roc_auc = roc_auc_score(y_test, LR_y_prob)
print(f'ROC-AUC Score: {LR_roc_auc:.2f}')


# In[57]:


LR_conf_matrix = confusion_matrix(y_test, LR_y_pred)
df_conf_matrix = pd.DataFrame(LR_conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print("Logistic Regresion Confusion Matrix:")
print(df_conf_matrix)


# In[58]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust hyperparameters as needed

# Train the model
rf_model.fit(X_train, y_train)


# In[59]:


# Predict class labels
rf_y_pred = rf_model.predict(X_test)

# Predict probabilities for ROC-AUC
rf_y_prob = rf_model.predict_proba(X_test)[:, 1]  # Assuming binary classification; for multiclass, adjust accordingly


# In[60]:


# Classification Report
print("Classification Report:")
print(classification_report(y_test, rf_y_pred))


# In[61]:


# ROC-AUC Score
rf_roc_auc = roc_auc_score(y_test, rf_y_prob)
print(f'ROC-AUC Score: {rf_roc_auc:.2f}')


# In[62]:


# Confusion Matrix
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
df_conf_matrix = pd.DataFrame(rf_conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print("Random Forest Confusion Matrix:")
print(df_conf_matrix)


# In[63]:


dnn_model = Sequential()
dnn_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
dnn_model.add(Dense(64, activation='relu'))
dnn_model.add(Dense(32, activation='relu'))
dnn_model.add(Dense(1, activation='sigmoid'))  # For binary classification; use 'softmax' for multiclass


# In[64]:


# Compile the model
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC()])


# In[65]:


# Train the model
history = dnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)


# In[66]:


# Generate predictions
dnn_y_pred = (dnn_model.predict(X_test) > 0.5).astype(int)
dnn_y_prob = dnn_model.predict(X_test).ravel()  # Flatten array for binary classification


# In[67]:


# Classification Report
print("Classification Report:")
print(classification_report(y_test, dnn_y_pred))


# In[68]:


# ROC-AUC Score
dnn_roc_auc = roc_auc_score(y_test, dnn_y_prob)
print(f'ROC-AUC Score: {dnn_roc_auc:.2f}')


# In[69]:


dnn_conf_matrix = confusion_matrix(y_test, dnn_y_pred)
df_conf_matrix = pd.DataFrame(dnn_conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print("Deep learning Confusion Matrix:")
print(df_conf_matrix)


# In[70]:


# Define predictions and true values (Replace these with your actual data)
models = {
    'Logistic Regresion': LR_model,
    'Random Forest': rf_model
}

y_probs = {
    'Logistic Regresion': LR_y_prob,
    'Random Forest': rf_y_prob
}

# Plot ROC Curves
def plot_roc_curves(models, y_test, y_probs):
    plt.figure(figsize=(10, 6))
    
    for label, y_prob in y_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve ({label}) (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# Plot Confusion Matrices
def plot_confusion_matrices(models, X_test, y_test):
    for label, model in models.items():
        y_pred = model.predict(X_test)  # Get predictions from the model
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix for {label}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
# Call the functions with your data
plot_roc_curves(models, y_test, y_probs)
plot_confusion_matrices(models, X_test, y_test)


# In[71]:


plt.figure(figsize=(8, 6))
sns.heatmap(dnn_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix for Deep learning')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[72]:


def compute_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    return metrics


# In[73]:


def prepare_metrics_df(metrics_dict):
    df = pd.DataFrame(metrics_dict).T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Model'}, inplace=True)
    return df


# In[74]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report

y_pred_model1 = LR_y_pred
y_pred_model2 = rf_y_pred
y_pred_model3 = dnn_y_pred

# Compute metrics
metrics_model1 = compute_metrics(y_test, y_pred_model1)
metrics_model2 = compute_metrics(y_test, y_pred_model2)
metrics_model3 = compute_metrics(y_test, y_pred_model3)

# Prepare metrics for plotting
metrics_dict = {
    'Logistic Regresion': metrics_model1,
    'Random Forest': metrics_model2,
    'Deep learning': metrics_model3
}
metrics_df = prepare_metrics_df(metrics_dict)

# Plot the metrics
plt.figure(figsize=(12, 8))

# Plot Accuracy, Precision, Recall, and F1-Score
metrics_df.set_index('Model').plot(kind='bar', figsize=(12, 8), rot=45)
plt.title('Comparison of Models')
plt.ylabel('Score')
plt.xlabel('Model')
plt.ylim(0, 1)  # Assuming all metrics are normalized between 0 and 1
plt.legend(loc='best')
plt.grid(axis='y')
plt.show()


# In[ ]:




