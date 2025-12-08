import pandas as pd
# import the dataset into pandas DataFrames
df_training = pd.read_csv('../../data/unsw-nb15/raw/UNSW_NB15_training-set.csv')
df_testing = pd.read_csv('../../data/unsw-nb15/raw/UNSW_NB15_testing-set.csv')

# stack the training and testing sets
df_data = pd.concat([df_training, df_testing], axis=0)

# remove the columns 'id' and 'attack_cat'
df_data.drop('id', inplace=True, axis=1)
df_data.drop('attack_cat', inplace=True, axis=1)

# 'is_ftp_login' should be a binary feature, we remove the instances that hold the values 2 and 4
df_data = df_data[df_data['is_ftp_login'] != 2]
df_data = df_data[df_data['is_ftp_login'] != 4]

categorical_features = ['state', 'service', 'proto']
df_data = pd.get_dummies(df_data, columns=categorical_features, prefix=categorical_features, prefix_sep=":")
# move the labels back to the last column
df_data['labels'] = df_data.pop('label')

# Min-Max normalization on the non-binary features
# the min and max values are computed on the training set
continuous_features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst']
min = df_data[:df_training.shape[0]][continuous_features].min()
max = df_data[:df_training.shape[0]][continuous_features].max()
df_data[continuous_features] = (df_data[continuous_features] - min) / (max - min)

df_training = df_data[:df_training.shape[0]]    
df_testing = df_data[df_training.shape[0]:]

df_training.to_csv('../../data/unsw-nb15/processed/train.csv', index=False)
df_testing.to_csv('../../data/unsw-nb15/processed/test.csv', index=False)
