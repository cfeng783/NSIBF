class RandomizedGS(object):
    '''
    The utility class for hyperparameters tuning of ML models based on Randomized Grid Search.
    Note that Randomized Grid Search only support const and categorical hyperparameters, anything outside those
    two categories will be ignored!
    
    :param model: The ML model, should be an object of AbsModel 
    :param hyperparameters: the list of hyperparameters of the ML model
    :param train_x: the input data for training
    :param train_y: the output data for training
    :param neg_x: negative sampled data
    :param neg_y: negative sampled data labels

    '''


    def __init__(self, model, hyperparameters, train_x, train_y, neg_x,neg_y):
        '''
        Constructor
        '''
        self.model = model
        self.hyperparameters = hyperparameters
        
        self.train_x = train_x
        self.train_y = train_y
        self.neg_x = neg_x
        self.neg_y = neg_y
        
    
    def _eval(self, **args): 
        self.model.train(self.train_x,self.train_y,**args)
        score = self.model.score(self.neg_x,self.neg_y)
        
        if score > self.best_score:
            self.model.save_model()
            self.best_score = score
            self.best_config = args.copy()
        return score
            
    def run(self, n_searches ,verbose=1):
        """
        Run the randomized grid search algorithm to find the best hyperparameter configuration for the ML model
        
        :param n_searches: the number of searches
        :param verbose: higher level of verbose prints more messages during running the algorithm 
            (default is 1)
        :return model: the optimized model
        :return best_score: the best score achieved by the optimized model
        """
        self.best_score = float('-inf')
        
        candidates = []
        for _ in range(n_searches):
            param_list = []
            for hp in self.hyperparameters:
                param_list.append(hp.getValue())
            candidates.append(param_list) 
        
        for can in candidates:
            i = 0
            can_dict = {}
            for hp in self.hyperparameters:
                can_dict[hp.name] = can[i]
                i += 1
            score = self._eval(**can_dict)
            if verbose > 0:
                print('score:', score, can_dict)
        self.model.load_model()
        return self.model, self.best_config, self.best_score

