import numpy as np
from ...preprocessing import ContinousSignal,DiscreteSignal
from filterpy.kalman import unscented_transform,JulierSigmaPoints
from tensorflow import keras
from tensorflow.keras import layers
import tempfile
from scipy.linalg import cholesky
from ..base import BaseModel,DataExtractor,override
from . import nearestPD
from ...utils import reset_random_seed
from scipy.spatial.distance import mahalanobis
from sklearn import metrics


class NSIBF(BaseModel,DataExtractor):
    '''
    Neural system identification and bayesian filtering for anomaly detection.
    
    :param signals: the list of signals the model is dealing with
    :param input_range: the length of input sequence for covariate encoder
    :param window_length: the number of time points for stacking sensor measurements
    '''
    def __init__(self, signals, window_length, input_range):
        self.signals = signals
        self.wl = window_length
        self.input_range = input_range
        
        self.targets = []
        self.covariates = []
        for signal in self.signals:
            if signal.isInput==True:
                if isinstance(signal, ContinousSignal):
                    self.covariates.append(signal.name)
                if isinstance(signal, DiscreteSignal):
                    self.covariates.extend(signal.get_onehot_feature_names())
            if signal.isOutput==True:
                self.targets.append(signal.name)
        
        'mean vector of hidden states'
        self.z = None
        'covariance matrix of hidden states'
        self.P = None
        'process noise matrix'
        self.Q = None 
        'sensor measurement noise matrix'
        self.R = None
        'neural networks for state encoding'
        self.g_net = None
        'neural networks for state transition'
        self.f_net = None
        'neural networks for state decoding, aka measurement function'
        self.h_net = None
        
        self.loss_weights = [0.45,0.45,0.1]
    
    @override
    def extract_data(self, df_ori, freq=1, purpose='train', label=None):
        """
        Extract data from given dataframe
        
        :param df: the Pandas DataFrame containing the data 
        :param freq: the sampling frequency 
            (default is 1)
        :param purpose: {"train","predict","AD"}, the purpose of data extraction
            (default is "train")
        :param label: the name of the anomaly label column
            (defualt is None)
        :return x: the input target variables, matrix of shape = [n_samples, n_features]
        :return u: the input covariates, matrix of shape = [n_samples, input_range, n_features]
        :return y: the output target variables
            If purpose is 'train', y is matrix of shape = [n_samples, n_features]
            Otherwise, y is None
        :return z: the anomaly labels
            If label is not None, z is matrix of shape = [n_samples,window_length]
            Otherwise, z is None
        """
        
        df = df_ori.copy()
        x_feats,u_feats,y_feats,z_feats = [],[],[],[]
        
        for entry in self.targets:
            for i in range(1,self.wl+1):
                if i < self.wl:
                    j = self.wl-i
                    df[entry+'-'+str(j)] = df[entry].shift(j)
                    x_feats.append(entry+'-'+str(j))
                else:
                    x_feats.append(entry)  
            if purpose == 'train':
                for i in range(1,self.wl+1):
                    df[entry+'+'+str(i)] = df[entry].shift(-i)
                    y_feats.append(entry+'+'+str(i))
            
        for entry in self.covariates:
            for i in range(1,self.input_range+1):
                if i < self.input_range:
                    j = self.input_range-i
                    df[entry+'-'+str(j)] = df[entry].shift(j)
                    u_feats.append(entry+'-'+str(j))
                else:
                    u_feats.append(entry)
                    
        if label is not None:
            for i in range(1,self.wl+1):
                if i < self.wl:
                    j = self.wl-i
                    df[label+'-'+str(j)] = df[label].shift(j)
                    z_feats.append(label+'-'+str(j))
                else:
                    z_feats.append(label)       
        
        df = df.dropna(subset=x_feats+u_feats+y_feats+z_feats)
        df = df.reset_index(drop=True)
        
        if freq > 1:
            df = df.iloc[::freq,:]
            df = df.reset_index(drop=True)
        
        x = df.loc[:,x_feats].values
        
        if len(u_feats) > 0:
            u = df.loc[:,u_feats].values
            u = np.reshape(u, (len(u),len(self.covariates),self.input_range) )
            u = np.transpose(u,(0,2,1))
        else:
            u = None
        
        if label is None:
            z = None
        else:
            z = df.loc[:,z_feats].values
        
        if purpose=='train':
            y = df.loc[:,y_feats].values
            return x,u,y,z
        elif purpose == 'predict':
            return x,u,None,z
        elif purpose=='AD':
            return x,u,None,z
    
    
    def score_samples_via_residual_error(self,x,u):
        """
        get anomalies scores for samples via NSIBF-RECON and NSIBF-PRED
        
        :param x: the target variables, i.e., the measurements, matrix of shape = [n_timesteps, n_targets]
        :param u: the covariates, matrix of shape = [n_timesteps, input_range, n_feats]
        :return recon_scores: matrix of shape = [n_timesteps,]
        :return pred_scores: matrix of shape = [n_timesteps-1,]
        """
        (x_recon,x_pred,_) = self.estimator.predict([x,u])
        
        recon_scores = np.mean(np.abs(x-x_recon),axis=1)

        pred_scores = np.mean(np.abs(x[:-1,:]-x_pred[1:,:]),axis=1)

        return recon_scores,pred_scores
    
    def score_samples(self, x, u, reset_hidden_states=True):
        """
        get anomalies scores for samples via Baysian filtering
        
        :param x: the target variables, i.e., the measurements, matrix of shape = [n_timesteps, n_targets]
        :param u: the covariates, matrix of shape = [n_timesteps, input_range, n_feats]
        :param reset_hidden_states: whether or not to reset the hidden states
            If True, the measurements in the first timestep will be used to initialize the hidden states
            Otherwise, the measurements in the first timestep will be ignored 
            (default is True)
        :return scores: the anomaly scores from the second timestep, matrix of shape = [n_timesteps-1,]
        """
        if self.Q is None or self.R is None:
            print('please estimate noise before running this method!')
            return None
        
        if reset_hidden_states:
            self.z = self._encoding(x[0,:])
            self.P = np.diag([0.00001]*len(self.z))
        
        anomaly_scores = []
        for t in range(1,len(x)):
            print(t,'/',len(x))
            u_t = u[t-1,:,:]
            x_t = x[t,:]
            
            x_mu,x_cov = self._bayes_update(x_t, u_t)
        
            inv_cov = np.linalg.inv(x_cov)
            score = mahalanobis(x[t,:], x_mu, inv_cov)
            anomaly_scores.append(score)
        
        return np.array(anomaly_scores)
    
  
    def estimate_noise(self,x,u,y):
        """
        Estimate the sensor and process noise matrices from given data
        
        :param x: the input data for targets, matrix of shape = [n_timesteps, n_targets]
        :param u: the input data for covariates, matrix of shape = [n_timesteps, input_range, n_covariates]
        :param y: the output data for targets, matrix of shape = [n_timesteps, n_targets]
        :return self
        """
        s = self.g_net.predict(x)
        s_next_true = self.g_net.predict(y)
        s_next_pred = self.f_net.predict([s,u])
        self.Q = np.cov(np.transpose(s_next_pred-s_next_true))
        
        x_pred = self.h_net.predict(s)
        self.R = np.cov(np.transpose(x_pred-x))
        return self
    
    def _encoding(self,x):
        x = np.array([x]).astype(np.float)
        z = self.g_net.predict(x)
        return z[0,:]
        
    def _state_transition_func(self,z,u):        
        U = np.array([u]*len(z))
        X = [z, U]
        z_next = self.f_net.predict(X)
        return z_next 
    
    def _measurement_func(self,z):
        y = self.h_net.predict(z)
        return y
    
    def _sqrt_func(self,x):
        try:
            result = cholesky(x)
        except:
            result = np.linalg.cholesky(nearestPD(x))
        return result
    
    
    def _bayes_update(self,x_t,u_t):
        'Prediction step'
        points = JulierSigmaPoints(n=len(self.z),kappa=3-len(self.z),sqrt_method=self._sqrt_func)
        sigmas = points.sigma_points(self.z, self.P)
#         print(sigmas.shape)
        sigmas_f = self._state_transition_func(sigmas,u_t)
        z_hat, P_hat = unscented_transform(sigmas_f,points.Wm,points.Wc,self.Q)
#             print('z_predict=',z_hat,'P_predict=',P_hat)
        
        'Update step'
        sigmas_h = self._measurement_func(sigmas_f)
#             print('sigmas_h',sigmas_h)
        x_hat, Px_hat = unscented_transform(sigmas_h,points.Wm,points.Wc,self.R)
#             print('x_predict=',x_hat)        
        Pxz = np.zeros((len(z_hat),len(x_hat)))
        for i in range(len(sigmas)):
            Pxz += points.Wc[i] * np.outer(sigmas_f[i]-z_hat,sigmas_h[i]-x_hat)
        
        try:
            K = np.dot(Pxz,np.linalg.inv(Px_hat))
        except:
            K = np.dot(Pxz,np.linalg.pinv(Px_hat))
        self.z = z_hat + np.dot(K,x_t-x_hat)
        self.P = P_hat - np.dot(K,Px_hat).dot(np.transpose(K))
        
        return x_hat, Px_hat
    
    def filter(self,x,u,reset_hidden_states=True):
        """
        Bayesian filtering
        
        :param x: the target variables, i.e., the measurements, matrix of shape = [n_timesteps, n_targets]
        :param u: the covariates, matrix of shape = [n_timesteps, input_range, n_covariates]
        :param reset_hidden_states: whether or not to reset the hidden states
            If True, the measurements in the first timestep will be used to initialize the hidden states
            Otherwise, the measurements in the first timestep will be ignored 
            (default is True)
        :return x_mu: the predicted mean of measurements from the Update timestep, matrix of shape = [n_timesteps-1, n_feats]
        :return x_cov: the predicted covariance of measurements from the Update timestep, matrix of shape = [n_timesteps-1, n_feats, n_feats]
       """
        
        if self.Q is None or self.R is None:
            print('please estimate noise before running this method!')
            return None
        
        if reset_hidden_states:
            self.z = self._encoding(x[0,:])
            self.P = np.diag([0.00001]*len(self.z))
        
        mu_x_list, cov_x_list = [],[]
        for t in range(1,len(x)):
            print(t,'/',len(x))
            u_t = u[t-1,:,:]
            x_t = x[t,:]
            
            x_hat,Px_hat = self._bayes_update(x_t, u_t)
            
            mu_x_list.append(x_hat)
            cov_x_list.append(Px_hat)
            
        return np.array(mu_x_list),np.array(cov_x_list)
    
    @override
    def predict(self,x,u):
        (x_recon,x_pred,_) = self.estimator.predict([x,u])
        return x_recon,x_pred
    
    @override
    def train(self, x, y, z_dim, hnet_hidden_layers=1, fnet_hidden_layers=1, fnet_hidden_dim=8, 
                uencoding_layers=1,uencoding_dim=8,z_activation='tanh', l2=0.0,
                optimizer='adam',batch_size=256,epochs=10,
                validation_split=0.2,save_best_only=True,verbose=0):
        """
        Build a neural network model for system identification according to the given hyperparameters, 
        and train the model using the given data
        
        :param x: the input data, it consists of two parts [x1,x2], 
                    x1 is the target variables in the current timestep, matrix of shape = [n_samples, n_targets]
                    x2 is the covariates in the input range, matrix of shape = [n_samples, input_range, n_covariates]
        :param y: the ground truth output data, it consists of two parts [x1,x2], 
                    y1 is the reconstructed target variables in the current timestep, matrix of shape = [n_samples, n_targets]
                    y2 is the predicted target variables in the next timestep, matrix of shape = [n_samples, n_targets]
        :param z_dim: the dimension of hidden embedding for target variables
        :param hnet_hidden_layers: number of hidden layers for h_net
            (default is 1)
        :param fnet_hidden_layers: number of hidden layers for f_net
            (default is 1)
        :param fnet_hidden_dim: number of hidden dimensions for f_net
            (default is 8)
        :param uencoding_layers: number of encoding layers for covariates
            (default is 1)
        :param uencoding_dim: number of hidden dimensions for uencoding_layers
            (default is 8)
        :param z_activation: the activation function for hidden embedding for target variables
            (default is 'tanh')    
        :param optimizer: the optimizer for gradient descent
            (default is 'adam')
        :param batch_size: the batch size
            (default is 256)
        :param epochs: the maximum epochs to train the model
            (default is 10)
        :param validation_split: the validation size when training the model
            (default is 0.2)
        :param save_best_only: save the model with best validation performance during training
            (default is True)
        :param verbose: 0 indicates silent, higher values indicate more messages will be printed
            (default is 0)
        :return self
        """
        z = np.zeros((x[0].shape[0],z_dim))
        x_dim, u_dim = x[0].shape[1], x[1].shape[2]
#         print(x[0].shape,x[1].shape)
        keras.backend.clear_session()
        reset_random_seed()
        model, g_net, h_net, f_net = self._make_network(x_dim, u_dim, z_dim, 
                                                          hnet_hidden_layers, fnet_hidden_layers, 
                                                          fnet_hidden_dim, uencoding_layers,uencoding_dim,
                                                          z_activation,l2)
        model.compile(optimizer=optimizer, loss=['mse','mse','mse'], loss_weights=self.loss_weights)

        if save_best_only:
            checkpoint_path = tempfile.gettempdir()+'/NSIBF.ckpt'
            cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_best_only=True, save_weights_only=True)                          
            model.fit(x, y+[z], batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[cp_callback], verbose=verbose)
            model.load_weights(checkpoint_path)
            g_net.compile(optimizer=optimizer, loss='mse')
            h_net.compile(optimizer=optimizer, loss='mse')
            f_net.compile(optimizer=optimizer, loss='mse')
        else:
            model.fit(x, y+[z], batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=verbose)
            g_net.compile(optimizer=optimizer, loss='mse')
            h_net.compile(optimizer=optimizer, loss='mse')
            f_net.compile(optimizer=optimizer, loss='mse')
        
        self.estimator = model
        self.g_net = g_net
        self.h_net = h_net
        self.f_net = f_net
        
        return self
    
    
    @override
    def score(self,neg_x,neg_y):
        """
        Score the model based on datasets with uniform negative sampling.
        Better score indicate a higher performance
        For efficiency, the AUC score of NSIBF-RECON is used for scoring in this version.
        Since normal samples are minority, we calculate 1-auc_score.
        """
        
        recon_scores,_ = self.score_samples_via_residual_error(neg_x[0],neg_x[1])
        t1 = metrics.roc_auc_score(neg_y, recon_scores)
        return 1-t1
    
    @override
    def save_model(self,model_path=None):
        """
        save the model to files
        
        :param model_path: the target folder whether the model files are saved (default is None)
            If None, a tempt folder is created
        """
        
        if model_path is None:
            model_path = tempfile.gettempdir()
        
        self.estimator.save(model_path+'/NSIBF.h5',save_format='h5')
        self.f_net.save(model_path+'/NSIBF_f.h5',save_format='h5')
        self.g_net.save(model_path+'/NSIBF_g.h5',save_format='h5')
        self.h_net.save(model_path+'/NSIBF_h.h5',save_format='h5')
    
    @override
    def load_model(self,model_path=None):
        """
        load the model from files
        
        :param model_path: the target folder whether the model files are located (default is None)
            If None, load models from the tempt folder
        :return self
        """
        if model_path is None:
            model_path = tempfile.gettempdir()
        self.estimator = keras.models.load_model(model_path+'/NSIBF.h5')
        self.f_net = keras.models.load_model(model_path+'/NSIBF_f.h5')
        self.g_net = keras.models.load_model(model_path+'/NSIBF_g.h5')
        self.h_net = keras.models.load_model(model_path+'/NSIBF_h.h5')
        
        return self
    
    
    def _make_network(self, x_dim, u_dim, z_dim, 
                      hnet_hidden_layers, fnet_hidden_layers, 
                      fnet_hidden_dim, uencoding_layers,uencoding_dim,z_activation,l2):
        x_input = keras.Input(shape=(x_dim),name='x_input')
        u_input = keras.Input(shape=(self.input_range,u_dim),name='u_input')
        z_input = keras.Input(shape=(z_dim),name='z_input')
        
        interval = (x_dim-z_dim)//(hnet_hidden_layers+1)
        hidden_dims = []
        hid_dim = max(1,x_dim-interval)
        hidden_dims.append(hid_dim)
        g_dense1 = layers.Dense(hid_dim, activation='relu',name='g_dense1')(x_input)
        for i in range(1,hnet_hidden_layers):
            hid_dim = max(1,x_dim-interval*(i+1))
            if i == 1:
                g_dense = layers.Dense(hid_dim, activation='relu') (g_dense1)
            else:
                g_dense = layers.Dense(hid_dim, activation='relu') (g_dense)
            hidden_dims.append(hid_dim)
        if hnet_hidden_layers > 1:
            g_out = layers.Dense(z_dim, activation=z_activation,name='g_output',activity_regularizer=keras.regularizers.l2(l2))(g_dense)
        else:
            g_out = layers.Dense(z_dim, activation=z_activation,name='g_output',activity_regularizer=keras.regularizers.l2(l2))(g_dense1)
        g_net = keras.Model(x_input,g_out,name='g_net')
         
        h_dense1 = layers.Dense(hidden_dims[len(hidden_dims)-1], activation='relu',name='h_dense1')(z_input)
        for i in range(1,hnet_hidden_layers):
            if i == 1:
                h_dense = layers.Dense(hidden_dims[len(hidden_dims)-1-i], activation='relu') (h_dense1)
            else:
                h_dense = layers.Dense(hidden_dims[len(hidden_dims)-1-i], activation='relu') (h_dense)
        
        if hnet_hidden_layers > 1:
            h_out = layers.Dense(x_dim, activation='linear',name='h_output') (h_dense)
        else:
            h_out = layers.Dense(x_dim, activation='linear',name='h_output') (h_dense1)
        h_net = keras.Model(z_input,h_out,name='h_net')
         
        if uencoding_layers == 1:
            f_uencoding = layers.LSTM(uencoding_dim, return_sequences=False)(u_input)
        else:
            f_uencoding = layers.LSTM(uencoding_dim, return_sequences=True)(u_input)
            for i in range(1,uencoding_layers):
                if i == uencoding_layers-1:
                    f_uencoding = layers.LSTM(uencoding_dim, return_sequences=False)(f_uencoding)
                else:
                    f_uencoding = layers.LSTM(uencoding_dim, return_sequences=True)(f_uencoding)
        f_concat = layers.Concatenate(name='f_concat')([z_input,f_uencoding])
        f_dense = layers.Dense(fnet_hidden_dim, activation='relu')(f_concat)
        for i in range(1,fnet_hidden_layers):
            f_dense = layers.Dense(fnet_hidden_dim, activation='relu') (f_dense)
        f_out = layers.Dense(z_dim, activation=z_activation,name='f_output',activity_regularizer=keras.regularizers.l2(l2)) (f_dense)
        f_net = keras.Model([z_input,u_input],f_out,name='f_net')
 
        z_output = g_net(x_input)
        x_output = h_net(z_output)
        z_hat_output= f_net([z_output,u_input])
        x_hat_output = h_net(z_hat_output)
        smoothing = layers.Subtract(name='smoothing')([z_output,z_hat_output])
        model = keras.Model([x_input,u_input],[x_output,x_hat_output,smoothing])
         
        return model, g_net,h_net,f_net
    
    