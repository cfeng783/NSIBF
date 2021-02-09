import numpy as np
from framework.preprocessing.data_loader import get_simulation_data
from framework.models import NSIBF
from sklearn import metrics
import math
from framework.preprocessing import normalize_and_encode_signals
import matplotlib.pyplot as plt
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

train_df,test_df,signals = get_simulation_data()

seqL = 31
kf = NSIBF(signals,window_length=seqL, input_range=2*seqL)


train_df = normalize_and_encode_signals(train_df,signals,scaler='min_max') 
x,u,y,_ = kf.extract_data(train_df)
print(x.shape,u.shape,y.shape)
print(np.isnan(x).sum(),np.isnan(u).sum(),np.isnan(y).sum())
pos = len(x)*3//4
train_x = x[:pos,:]
train_u = u[:pos,:]
train_y = y[:pos,:]
x_train = [train_x,train_u]
y_train = [train_x,train_y]
val_x = x[pos:,:]
val_u = u[pos:,:]
val_y = y[pos:,:]

'uncomment to retrain model'
# kf.train(x_train, y_train, z_dim=2, hnet_hidden_layers=2, 
#          fnet_hidden_layers=2, fnet_hidden_dim=8, 
#          uencoding_layers=1, uencoding_dim=4, l2=0.01, 
#          epochs=50, validation_split=0.1, save_best_only=True, verbose=2)
# kf.save_model('../results/qualitative')
   
kf = kf.load_model('../results/qualitative')
kf.estimate_noise(val_x,val_u,val_y)

test_df = normalize_and_encode_signals(test_df,signals,scaler='min_max')
test_x,test_u,_,labels = kf.extract_data(test_df,purpose='AD',freq=seqL,label='label')
    
x_mu,x_cov = kf.filter(test_x, test_u,reset_hidden_states=True)
x_recon,x_pred = kf.predict(test_x,test_u)


ls = []
true_x = []
mu_x = []
std_x = []
rec_x = []
pred_x = []
for i in range(len(x_mu)):
    for j in range(seqL):
        mu_x.append(x_mu[i,j])
        std_x.append(math.sqrt(x_cov[i,j,j]))
        rec_x.append(x_recon[i+1,j])
        pred_x.append(x_pred[i,j])
        true_x.append(test_x[i+1,j])
        ls.append(labels[i+1,j])
mu_x,std_x = np.array(mu_x),np.array(std_x)

T = np.linspace(1,len(true_x),len(true_x))
plt.plot(T[1000:1300],true_x[1000:1300],linestyle='-',label='Observed measurements')
plt.plot(T[1000:1300],rec_x[1000:1300],linestyle='-.',label='NSIBF-RECON')
plt.plot(T[1000:1300],pred_x[1000:1300],linestyle=':',label='NSIBF-PRED')
plt.plot(T[1000:1300],mu_x[1000:1300],linestyle='--',label='NSIBF mean')
plt.fill_between(T[1000:1300], mu_x[1000:1300]-std_x[1000:1300], mu_x[1000:1300]+std_x[1000:1300], alpha=0.5,color='y',label='NSIBF one std interval')
_,_,ymin,ymax = plt.axis()
anomaly_idx = [i for i in range(1000,1300) if ls[i]==1]
plt.fill_between(anomaly_idx,ymin,ymax,alpha=0.75,color='grey',label='Anomalous period')
plt.legend(loc=1)
plt.show()

labels = labels.sum(axis=1)
labels[labels>0]=1
z_scores = kf.score_samples(test_x, test_u,reset_hidden_states=True)
recon_scores,pred_scores = kf.score_samples_via_residual_error(test_x,test_u)


z_scores = np.nan_to_num(z_scores)
T = np.linspace(1,len(test_x)*seqL,len(test_x)-1)
colors = ['r','g','b','c','m','y','k']
dim = 4
for j in range(dim):
    ax = plt.subplot(dim,1,j+1)
    if j == 0:
        plt.plot(T,labels[1:],label='anomaly label',color=colors[j%7])
        ax.xaxis.set_ticklabels([])
    elif j == 1:
        plt.plot(T,recon_scores[1:],label='NSIBF-RECON',color=colors[j%7])
        plt.ylabel('residual error')
        ax.xaxis.set_ticklabels([])
    elif j == 2:
        plt.plot(T,pred_scores,label='NSIBF-PRED',color=colors[j%7])
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        plt.ylabel('residual error')
        ax.xaxis.set_ticklabels([])
    elif j == 3:
        plt.plot(T,z_scores,label='NSIBF',color=colors[j%7])
        plt.ylabel('anomaly score')
    plt.legend()
      
plt.show()
  
plt.plot([0,1],[0,1],'k--')
fpr,tpr,thresholds = metrics.roc_curve(labels[1:],recon_scores[1:],pos_label=1)
plt.plot(fpr,tpr,label='NSIBF-RECON')
fpr,tpr,thresholds = metrics.roc_curve(labels[1:],pred_scores,pos_label=1)
plt.plot(fpr,tpr,label='NSIBF-PRED')
fpr,tpr,thresholds = metrics.roc_curve(labels[1:],z_scores,pos_label=1)
plt.plot(fpr,tpr,label='NSIBF')
  
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.show()
