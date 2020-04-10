from sklearn import ensemble

MODELS = {
    "RandomForest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2)
    # "ExtraTrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2)

}

# print(MODELS.keys("RandomForest"))
