import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class CollaborativeFiltering(object):
    def __init__(self, metric='euclidean', algorithm='auto', neighbor_ratio=0.001):
        self.metric = metric
        self.algorithm = algorithm
        self.neighbor_ratio = neighbor_ratio
        self.classifier = None
        self.num_train_users = -1

    def fit(self, X, Y):
        self.num_train_users = X.shape[0]
        # self.classifier = KNeighborsClassifier(metric=self.metric, algorithm=self.algorithm, n_neighbors=int(self.num_train_users * self.neighbor_ratio))
        self.classifier = KNeighborsClassifier(metric=self.metric, algorithm=self.algorithm, n_neighbors=int(16))

        # These products are during training not seen as target and thus cannot 
        # be learned to be predicted by the classifier
        # 
        self.num_products = Y.shape[1]
        all_indices = np.arange(self.num_products)
        indices_to_ignore = np.where(np.sum(Y, axis=0) == 0)[0]
        self.predictable_indices = all_indices[~np.isin(all_indices, indices_to_ignore)]

        # Only fit with the products that are in the training dataset targets, such that
        # we avoid issues with the return value of :predict_proba from sklearn
        # 
        self.classifier.fit(X, Y[:, self.predictable_indices])

    def predict_proba(self, X):
        y_pred = np.array(self.classifier.predict_proba(X))
        
        # Shape was (NumProducts, NumSamples, NumClasses), reshape to
        # (NumSamples, NumProducts, NumClasses)
        # 
        y_pred = np.swapaxes(y_pred, 0, 1)

        return  y_pred[:,:,1]
