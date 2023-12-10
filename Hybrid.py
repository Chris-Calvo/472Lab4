import ChrisMethodSVC
import ChrisMethodRF
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import JulianMethod

print()
print("\033[1m" + 'RF METHOD' + "\033[0m")

rf_model = RandomForestClassifier(n_estimators=100)
X_train, X_test, y_train, y_test = ChrisMethodRF.main()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print()

print("\033[1m" + 'SVC METHOD' + "\033[0m")
svc_model = SVC(kernel='linear')
X_train, X_test, y_train, y_test = ChrisMethodSVC.main()
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)
print()

print("\033[1m" + 'SIFT METHOD' + "\033[0m")
j_pred = JulianMethod.main()
print()

final_predictions = []
for rf_pred, svc_pred, j_pred in zip(rf_pred, svc_pred, j_pred):
    final_pred = 1 if rf_pred + svc_pred + j_pred >= 2 else 0  # Majority vote
    final_predictions.append(final_pred)

print("\033[1m" + 'FINAL CALCULATIONS' + "\033[0m")
cm = confusion_matrix(y_test, final_predictions)
tn, fp, fn, tp = cm.ravel()
frr = fn / (fn + tp)
far = fp / (fp + tn)

print(frr)
print(far)

'''  Note - 
min, max of FRR & FAR were calculated outside of this program (each individual program's min, max FAR & FRR were averaged)
EER was calulated outside of this program
'''