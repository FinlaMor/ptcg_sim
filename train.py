import lightgbm as lgb

X_train = [] #rows of features
y_train = [] # 1 = won game, 0 = Lost game

train_data = lgb.Dataset(X_train, label=y_train)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt'
}

num_round = 100
bst = lgb.train(params, train_data, num_round)

bst.save_model('pok_model.txt')
print('Model trained and saved')