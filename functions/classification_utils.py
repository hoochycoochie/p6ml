from functions import text_utils
from sklearn.metrics import roc_curve,auc,adjusted_rand_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
import numpy as np
 




def grid_search_cv_multiclass(model,param_grid,scoring,cv,label_encoder,all_labels,X_train,X_test,y_train,y_test,title1,title2,title3):
    gr=GridSearchCV(model,cv=cv,param_grid=param_grid,scoring=scoring)
    gr.fit(X_train,y_train)
    classes=gr.best_estimator_.classes_

    y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))

    pred_prob = gr.best_estimator_.predict_proba(X_test)
 
    labels=gr.best_estimator_.predict(X_test)
    #print('label1',labels)
    labels_inverse = np.array(
        list(map(lambda label: label_encoder.inverse_transform([label])[0], labels)
        )
        )
    #print('label2',labels)
    fpr = {}
    tpr = {}
    thresh ={}
    roc_auc = dict()

    n_class = classes.shape[0]
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], pred_prob[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])          
        name='%s vs Rest (AUC=%0.2f)'%(label_encoder.inverse_transform([classes[i]])[0],roc_auc[i])
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], name=name, mode='lines'))

    fig.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=1000, height=800
)
    fig.show()
    mean_auc=0
    for val in roc_auc.values():
        mean_auc += val
    mean_auc = mean_auc / len(roc_auc)
    ## Ajout graphique de clusterization
    ARI = np.round(adjusted_rand_score(y_test, labels),4)

    text_utils.TSNE_visu_fct(X_test,y_test,all_labels,labels,ARI,title1,title2,title3,labels_inverse)
    print('ARI',ARI)
    print('mean_auc',mean_auc)
    return mean_auc,ARI,gr
