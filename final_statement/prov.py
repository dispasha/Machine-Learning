import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import datetime
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data_train = pd.read_csv('features.csv', index_col='match_id')
data_test = pd.read_csv('features_test.csv', index_col='match_id')

train_Y = data_train['radiant_win']
columns_train_difference = data_train.columns.difference(
    data_test.columns.values.tolist()).tolist()
data_train.drop(columns_train_difference, axis=1, inplace=True)  # СѓРґР°Р»СЏРµРј РІРЅСѓС‚СЂРё РґР°С‚Р°СЃРµС‚Р°

train_size = len(data_train)
print("Select count=%s" % train_size)
for col in data_train.columns.values.tolist():
    count = data_train[col].count()
    if count != train_size:
        print("Column %s, len=%s" % (col, count))


verbose = 1

idx_split = data_train.shape[0]
data_full = pd.concat(
    [data_train, data_test])


for col in data_full.columns.values.tolist():
    maxVal = data_full.loc[data_full[col].notnull(), col].max() ** 2
    data_full.loc[data_full[col].isnull(), col] = maxVal

kf = KFold(n_splits=5, shuffle=True)

for est in range(10, 31, 10):
    clf = GradientBoostingClassifier(n_estimators=est,
                                     random_state=241)
    start_time = datetime.datetime.now()
    clf.fit(data_full.iloc[:idx_split, :], train_Y)

    scores = cross_val_score(clf, data_full.iloc[:idx_split, :], train_Y, scoring='roc_auc', cv=kf)
    val = round(scores.mean() * 100, 2)


param_grid = {'n_estimators': [60, 70], 'max_depth': range(3, 5),
              'max_features': ["log2"]}
clf_grid = GridSearchCV(GradientBoostingClassifier(random_state=2345), param_grid, cv=kf, n_jobs=1, verbose=verbose,
                        scoring='roc_auc')
clf_grid.fit(data_full.iloc[:idx_split, :], train_Y)
print("best_params")
print(clf_grid.best_params_)
print("best_score")
print(clf_grid.best_score_)

clf = GradientBoostingClassifier(
    **clf_grid.best_params_)
start_time = datetime.datetime.now()
clf.fit(data_full.iloc[:idx_split, :], train_Y)
print('Time elapsed:', datetime.datetime.now() - start_time)

scores = cross_val_score(clf, data_full.iloc[:idx_split, :], train_Y, scoring='roc_auc', cv=kf)
val = round(scores.mean() * 100, 2)
print("РћС†РµРЅРєР° РєР°С‡РµСЃС‚РІР° РїСЂРё СѓРІРµР»РёС‡РµРЅРёРё С‡РёСЃР»Р° РґРµСЂРµРІСЊРµРІ=%s" % val)


featureImportances = pd.DataFrame(data=clf.feature_importances_)
featureImportances.sort_values([0], ascending=False, inplace=True)
listCol = data_full.columns.values.tolist()

count = 1
for i in featureImportances.index:
    if featureImportances.loc[i][0] < 0.02: break
    count += 1

data_full = pd.concat(
    [data_train, data_test])
data_full.index = range(0, len(data_full))
del data_train, data_test
data_full.fillna(0, method=None, axis=1, inplace=True)

param_grid = {'C': np.logspace(-3, -1, 10)}


def getScoreLogisticRegression(text, data_train, saveToFile=False):
    clf_grid = GridSearchCV(LogisticRegression(random_state=241, n_jobs=-1), param_grid, cv=kf, n_jobs=1,
                            verbose=verbose, scoring='roc_auc')
    clf_grid.fit(data_full.iloc[:idx_split, :], train_Y)

    lr = LogisticRegression(n_jobs=-1, random_state=241,
                            **clf_grid.best_params_)
    lr.fit(data_train.iloc[:idx_split, :], train_Y)
    s = cross_val_score(lr, data_train.iloc[:idx_split, :], train_Y, scoring='roc_auc', cv=kf)  # РћС†РµРЅРєР° Р°Р»РіРѕСЂРёС‚РјР°
    sos = round(s.mean() * 100, 2)
    print("РћС†РµРЅРєР° РєР°С‡РµСЃС‚РІР° GridSearchCV (%s)=%s" % (text, sos))

    y_pred = pd.DataFrame(data=lr.predict_proba(data_train.iloc[idx_split:, :]))
    y_pred.sort_values([0], inplace=True)
    print(u'min=', y_pred.iloc[0, 1], '; max=',
          y_pred.iloc[y_pred.shape[0] - 1, 1])

    if saveToFile:
        y_pred.sort_index(inplace=True)
        y_pred.to_csv('Radiant win predict', columns=[1], index_label=['match_id'], header=['prediction'])


getScoreLogisticRegression("without scaling", data_full)

data_full_norm = pd.DataFrame(data=StandardScaler().fit_transform(data_full))
getScoreLogisticRegression("with scaling", data_full_norm)

del data_full_norm
cols = ['r%s_hero' % i for i in range(1, 6)] + ['d%s_hero' % i for i in range(1, 6)]
cols.append('lobby_type')

data_full_norm = pd.DataFrame(data=StandardScaler().fit_transform(data_full.drop(cols, axis=1)))
getScoreLogisticRegression("drop categories, with scaling", data_full_norm)

iid = pd.Series(data_full[cols].values.flatten()).drop_duplicates()
N = iid.shape[0]
iid = pd.DataFrame(data=list(range(N)), index=iid.tolist())
iid.sort_index(inplace=True)
print(u'СЃРєРѕР»СЊРєРѕ СЂР°Р·Р»РёС‡РЅС‹С… РёРґРµРЅС‚РёС„РёРєР°С‚РѕСЂРѕРІ РіРµСЂРѕРµРІ СЃСѓС‰РµСЃС‚РІСѓРµС‚ РІ РґР°РЅРЅРѕР№ РёРіСЂРµ: ', N)
print('РЎС‚Р°СЂС‚ dummy РєРѕРґРёСЂРѕРІР°РЅРёСЏ...')
start_time = datetime.datetime.now()

x_pick_d = pd.get_dummies(data_full[cols[5:]].astype('str'))
x_pick_r = pd.get_dummies(data_full[cols[:5]].astype('str'))
x_pick_r *= -1
x_pick_d.columns = [col[1:] for col in list(x_pick_d.columns)]
x_pick_r.columns = [col[1:] for col in list(x_pick_r.columns)]

x_pick = x_pick_d + x_pick_r
del x_pick_d, x_pick_r

print('Р—Р°РІРµСЂС€РёР»Рё. Time elapsed:', datetime.datetime.now() - start_time)

total = data_full_norm.join(x_pick, rsuffix='_',
                            how='inner')
del x_pick, data_full_norm
getScoreLogisticRegression("dummy coding", total, True)