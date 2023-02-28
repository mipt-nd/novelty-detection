# novelty-detection

- 3 sigma
- IForest
- One-SVM
- KNN
- Local Outlier Factor (LOF)
- K-means
- DBSCAN

**Personal suggestion on selecting an OD algorithm**. If you do not know which algorithm to try, go with:

- [ECOD](https://github.com/yzhao062/pyod/blob/master/examples/ecod_example.py): Example of using ECOD for outlier detection
- [Isolation Forest](https://github.com/yzhao062/pyod/blob/master/examples/iforest_example.py): Example of using Isolation Forest for outlier detection

They are both fast and interpretable. Or, you could try more data-driven approach [MetaOD](https://github.com/yzhao062/MetaOD).

**Outlier Detection with 5 Lines of Code:**

```python
# train an ECOD detector
from pyod.models.ecod import ECOD
clf = ECOD()
clf.fit(X_train)

# get outlier scores
y_train_scores = clf.decision_scores_ # raw outlier scores on the train data
y_test_scores = clf.decision_function(X_test) # predict raw outlier scores on test
```