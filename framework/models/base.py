from abc import ABC, abstractmethod

def override(f):
    return f

class BaseModel(ABC):
    """
    The base class for all models in predictive_maintenance
    """
    
    @abstractmethod
    def train(self, x, y=None, **params):
        """
        Create a model based on the give hyperparameters and train the model
        
        :param x: the input data
        :param y: the output data
            (default is None)
        :param params: the hyperparameters of the model
        :return self
        
        """
        pass
    
    @abstractmethod
    def predict(self, x):
        """
        Predict outputs for x
        
        :param x: the input data
        :return y: the output data 
        """
        pass
    
    @abstractmethod
    def score(self, x, y=None):
        """
        Score the model based on its performance on given data. 
        Higher score indicates better performance. 
        
        :param x: the input data
        :param y: the ground truth output data 
            (default is None)
        :return score: the score
        """
        pass
    
    @abstractmethod
    def save_model(self, model_path=None):
        """
        save the model to file
        
        :param model_path: path of the model, if it is None, a tempt file path shall be specified
            (default is None)
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path=None):
        """
        load the model from file
        
        :param model_path: path of the model, if it is None, a tempt file path shall be specified
            (default is None)
        :return self
        
        """
        pass


class DataExtractor(ABC):
    
    @abstractmethod
    def extract_data(self, df, freq=1, purpose='train', label=None):
        """
        Extract data from given dataframe
        
        :param df: the Pandas DataFrame containing the data 
        :param freq: the sampling frequency 
            (default is 1)
        :param purpose: {"train","predict","AD"}, the purpose of data extraction
            (default is "train")
        :param label: the name of the anomaly label column
            (defualt is None)
        """
        pass

        
    



    