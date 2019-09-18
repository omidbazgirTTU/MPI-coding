# SKlearn Libraries to use
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import time
# MPI Initialization
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
sum = 0
tmp = rank

# Creating the data
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

start_time = time.time()
#####

if comm.rank ==0:
    # SVM
    clf = SVC(gamma='auto')
    clf.fit(X, y) 
    print("SVM Prediction ",clf.predict([[-0.8, -1]]))
if comm.rank == 1 :
    # Random Forest
    RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    RF.fit(X, y)  
    print("RF Feature importance ",RF.feature_importances_)
    print("RF Prediction ", RF.predict([[-0.8, -1]]))
if comm.rank == 2 :
    # KNN
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y) 
    print("KNN Prediction ",neigh.predict([[-0.8, -1]]))
    print("KNN Probability ", neigh.predict_proba([[-0.8, -1]]))
if comm.rank > 2 :
    print("done")
