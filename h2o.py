import h2o
# Assumes that the cluster is up and running with the datasets
# We did not include it here as we created a Flow in H2O to read the files from disk and to run the respective scripts
def getAccuracy(m):
    f1Threshold = m.F1(xval=True)[0][0]
    accuracy = m.accuracy(xval = True, thresholds=[f1Threshold])
    print(m.algo, ': logloss:', m.logloss(xval = True), ', auc:', m.auc(xval=True), ', accuracy:', round(accuracy[0][1],4))
    print(m.confusion_matrix(xval=True, thresholds=[f1Threshold]))
    
    fn = 1
    tn = 2
    f1Threshold = 1
    thresholdAllowed = 0.05
    while fn/(tn+fn) > thresholdAllowed:
        confMtx = m.confusion_matrix(xval=True, thresholds=[f1Threshold]).to_list()
        fn = confMtx[1][0]
        tn = confMtx[1][1]
        #f1Threshold -= (f1Threshold * 0.1 + 0.0001)
        f1Threshold *= 0.9
        if f1Threshold <= 0.00002:
            break
#    print(f1Threshold, round(fn/(fn+tn)*100, 2), '% false negatives')
#    print(m.confusion_matrix(xval=True, thresholds=[f1Threshold]))
    bRate = str(round(confMtx[0][1] / (confMtx[0][0] + confMtx[0][1]) * 100, 1))
    mRate = str(round(confMtx[1][0] / (confMtx[1][0] + confMtx[1][1]) * 100, 1))
    return "At threshold of " + str(round(f1Threshold, 5))+", mRate of " + mRate + "%, bRate of " + bRate+"%"

h2o.connect(ip='xcnd14.comp.nus.edu.sg')

accuracy = []
for i in range(56):
    m = h2o.get_model('grid-dl_model_'+str(i))
    accuracy.append(getAccuracy(m))
#    print(round(m.auc(xval=True), 3), round(m.logloss(xval = True),3), round(m.accuracy(xval=True)[0][1],4))
#    print(i, m.logloss(xval = True), m.auc(xval = True))

for i in range(len(accuracy)):
    print(i, accuracy[i])
    
accuracy = []
for i in [52, 40, 36,24,48,54,44,28,20,53]:
    accuracy.append(getAccuracy(h2o.get_model('grid-dl_model_'+str(i))))

for i in range(len(accuracy)):
    print(i, accuracy[i])


h2o.connect(ip='xcnb0.comp.nus.edu.sg')

# GBM: 
# AUC: 0.9255, 
rf = h2o.get_model('DRF_0_AutoML_20180419_104304')
getAccuracy(rf)
accuracy = []
for i in range(3):
    m = h2o.get_model('GBM_grid_0_AutoML_20180419_104304_model_' + str(i))
    print(m.actual_params['distribution'])
    print(m.actual_params['ntrees'])
    print(m.actual_params['max_depth'])
    accuracy.append(getAccuracy(m))
for i in range(len(accuracy)):
    print(i, accuracy[i])
#    print(i)
#    print('!!', i, getAccuracy(h2o.get_model('GBM_grid_0_AutoML_20180419_104304_model_' + str(i))))
glm = h2o.get_model('GLM_grid_0_AutoML_20180419_104304_model_0')
getAccuracy(glm)
xrt = h2o.get_model('XRT_0_AutoML_20180419_104304')
getAccuracy(xrt)
