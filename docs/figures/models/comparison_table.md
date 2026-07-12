# Model comparison  -  full-grid runs (2026-07-02)

```
         model_name feature_set                           run_id  cv_roc_auc_mean  cv_roc_auc_std  test_accuracy  test_precision  test_recall  test_f1  test_roc_auc
logistic_regression    clinical 514db551010841ea9e4fe9fa2d0cfaf6           0.9195          0.0123         0.7667          0.7500       0.7500   0.7500        0.8817
logistic_regression         raw 182de41472bc48e58836ceeca238a7a0           0.9198          0.0230         0.7833          0.7778       0.7500   0.7636        0.8761
      random_forest         raw a27a3a154dd1404dbfbcd6c335fca914           0.9238          0.0149         0.7833          0.7586       0.7857   0.7719        0.8583
      random_forest    clinical 26f64de40d5145a4abcbe37a52d97933           0.9148          0.0206         0.7667          0.7917       0.6786   0.7308        0.8493
            xgboost    clinical 505a5c17bd8c4782b4b8420634260fe7           0.9313          0.0115         0.6833          0.6552       0.6786   0.6667        0.8415
            xgboost         raw 757115abc0e84e30a131b4728921afea           0.9242          0.0196         0.7500          0.6970       0.8214   0.7541        0.8315
```

Best run: logistic_regression / clinical (test ROC-AUC 0.8817, run_id 514db551010841ea9e4fe9fa2d0cfaf6)
