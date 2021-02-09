from abc import ABC,abstractmethod
import random
from enum import Enum

class HyperparameterType(Enum):
    UniformInteger = 301
    UniformFloat = 302
    Categorical = 303
    Const = 304

class baseHyperparameter(ABC):
    """
    The base class for Hyperparameters
    
    :param name: the name of the hyperparameter 
    :param hp_type: the type of the hyperparameter
    
    """


    def __init__(self,name,hp_type):
        '''
        Constructor
        '''
        self.name = name
        self.hp_type = hp_type
    
    @abstractmethod
    def getValue(self):
        """
        Get a value of the hyperparameter
        :return value: the value
        """
        pass
    
    @abstractmethod
    def getAllValues(self):
        """
        Get all possible values for the hyperparameter
        :return values: the all possible values
        """
        pass


class UniformIntegerHyperparameter(baseHyperparameter):
    '''
    The uniform integer hyperparameter class
    
    :param name: the name of the hyperparameter 
    :param lb: the lower bound of the hyperparameter
    :param lb: the upper bound of the hyperparameter
    '''


    def __init__(self, name, lb, ub):
        '''
        Constructor
        '''
        self.bot = lb
        self.top = ub
        super().__init__(name,HyperparameterType.UniformInteger)
    
    def getValue(self):
        """
        Get a random value of the hyperparameter
        :return value: the value
        """
        return random.randint(self.bot,self.top)
    
    def getAllValues(self):
        """
        Get all possible values for the hyperparameter
        :return values: a tuple including the lower bound and upper bound of the hyperparameter
        """
        return (self.bot,self.top)

class UniformFloatHyperparameter(baseHyperparameter):
    '''
    The uniform float hyperparameter class
    
    :param name: the name of the hyperparameter 
    :param lb: the lower bound of the hyperparameter
    :param lb: the upper bound of the hyperparameter
    '''


    def __init__(self, name, lb, ub):
        '''
        Constructor
        '''
        self.bot = lb
        self.top = ub
        super().__init__(name,HyperparameterType.UniformFloat)
    
    def getValue(self):
        """
        Get a random value of the hyperparameter
        :return value: the value
        """
        return random.uniform(self.bot,self.top)
    
    def getAllValues(self):
        """
        Get all possible values for the hyperparameter
        :return values: a tuple including the lower bound and upper bound of the hyperparameter
        """
        return (self.bot,self.top)
        

class CategoricalHyperparameter(baseHyperparameter):
    '''
    The categorical hyperparameter class
    
    :param name: the name of the hyperparameter 
    :param value_list: the list of all possible values of the hyperparameter
    '''


    def __init__(self, name, value_list):
        '''
        Constructor
        '''
        self.value_list = value_list
        super().__init__(name,HyperparameterType.Categorical)
    
    def getValue(self):
        """
        Get a random value of the hyperparameter
        :return value: the value
        """
        idx = random.randint(0,len(self.value_list)-1)
        return self.value_list[idx]
    
    def getAllValues(self):
        """
        Get all possible values for the hyperparameter
        :return values: the value list
        """
        return self.value_list


class ConstHyperparameter(baseHyperparameter):
    '''
    The constant hyperparameter class
    
    :param name: the name of the hyperparameter 
    :param value: the value of the hyperparameter
    '''


    def __init__(self, name, value):
        '''
        Constructor
        '''
        self.value = value
        super().__init__(name,HyperparameterType.Const)
    
    def getValue(self):
        """
        Get the value of the hyperparameter
        :return value: the value
        """
        return self.value
    
    def getAllValues(self):
        """
        Get all possible values for the hyperparameter
        :return values: the list only consists the value
        """
        return [self.value]  
        