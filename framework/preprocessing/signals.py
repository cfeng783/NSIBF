from enum import Enum

class SignalSource(Enum):
    sensor = 101
    controller = 102
    other = 103


class BaseSignal(object):
    '''
    The base signal class
    
    :param name: the name of the signal
    :param source: the source of the signal, can be SignalSource.sensor, SignalSource.controller or SignalSource.other
    :param isInput: whether it is an input signal
    :param isOutput: whether it is an output signal
    :param measure_point: the measurement point of a sensor, two sensors can share one measurement point for redundancy. If set to None, then the measurement point will be set to the name of the signal
        (default is None)
    '''


    def __init__(self, name, source, isInput, isOutput, measure_point=None):
        '''
        Constructor
        '''
        self.name = name
        self.source = source
        self.isInput = isInput
        self.isOutput = isOutput
        if measure_point is None:
            self.measure_point = name
        else:
            self.measure_point = measure_point
        



class ContinousSignal(BaseSignal):
    '''
    The class for signals which take continuous values
    
    :param name: the name of the signal
    :param source: the source of the signal, can be SignalSource.sensor, SignalSource.controller or SignalSource.other
    :param isInput: whether it is an input signal
    :param isOutput: whether it is an output signal
    :param min_value: minimal value for the signal
        (default is None)
    :param max_value: the maximal value for the signal
        (default is None)
    :param mean_value: mean for the signal value distribution
        (default is None)
    :param std_value: std for the signal value distribution
        (default is None)
    '''


    def __init__(self, name, source, isInput, isOutput, min_value=None, max_value=None, mean_value=None, std_value=None):
        '''
        Constructor
        '''
        super().__init__(name, source, isInput, isOutput)
        self.min_value = min_value
        self.max_value = max_value
        self.mean_value = mean_value
        self.std_value = std_value


class DiscreteSignal(BaseSignal):
    '''
    The class for signals which take discrete values
    
    :param name: the name of the signal
    :param source: the source of the signal, can be SignalSource.sensor, SignalSource.controller or SignalSource.other
    :param isInput: whether it is an input signal
    :param isOutput: whether it is an output signal
    :param values: the list of possible values for the signal
    '''


    def __init__(self, name, source, isInput, isOutput, values):
        '''
        Constructor
        '''
        super().__init__(name, source, isInput, isOutput)
        self.values = values
    
    def get_onehot_feature_names(self):
        """
        Get the one-hot encoding feature names for the possible values of the signal
        
        :return name_list: the list of one-hot encoding feature names
        """
        name_list = []
        for value in self.values:
            name_list.append(self.name+'='+str(value))
        return name_list
    
    def get_feature_name(self,value):
        """
        Get the one-hot encoding feature name for a possible value of the signal
        
        :param: value: a possible value of the signal
        :return name: the one-hot encoding feature name of the given value
        """
        return self.name+'='+str(value)
        