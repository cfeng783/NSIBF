from .signals import DiscreteSignal,ContinousSignal

def normalize_and_encode_signals(df_ori,signals,scaler=None):
    """
    Normalize and onehot encode the signals in the dataset
    
    :param df: the pandas DataFrame of the sensor dataset
    :param signals: the signals of interest in the dataset
    :param scaler: the scaler for normalization; 
                scaler=None indicates no normalization; 
                scaler='min_max' indicates using MinMax scaler;
                scaler='standard' indicates using standard scaler;
    :return modified_df: the pandas DataFrame of the sensor dataset in which the signals have been normalized or onehot encoded 
    """
    df = df_ori.copy()
    onehot_entries = {}
    'onehot encoding and normalisation'
    for signal in signals:
        if isinstance(signal, DiscreteSignal):
            onehot_entries[signal.name] = signal.get_onehot_feature_names()
            for value in signal.values:
                new_entry = signal.get_feature_name(value)
                df[new_entry] = 0
                df.loc[df[signal.name]==value,new_entry] = 1
        if isinstance(signal, ContinousSignal):
            df[signal.name] = df[signal.name].astype(float)
            if scaler == 'min_max':
                if signal.max_value is None or signal.min_value is None:
                    print('please specify min max values for signal',signal.name)
                if signal.max_value != signal.min_value:
                    df[signal.name]=df[signal.name].apply(lambda x:float(x-signal.min_value)/float(signal.max_value-signal.min_value))
            elif scaler == 'standard':
                if signal.mean_value is None or signal.std_value is None:
                    print('please specify mean and std values for signal',signal.name)
                if signal.std_value != 0:
                    df[signal.name]=df[signal.name].apply(lambda x:float(x-signal.mean_value)/float(signal.std_value))
    return df