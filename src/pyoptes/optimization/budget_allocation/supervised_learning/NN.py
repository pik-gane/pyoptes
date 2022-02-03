from torch import nn

class RNNetwork(nn.Module): #Recurrent Neural Network
    """Network with recurrent layers and ReLU activations."""    
    def __init__(self, in_features: int, out_features: int, bias: bool): 
        super(RNNetwork, self).__init__()
            
        """activation functions"""
        #self.act_func = nn.ReLU()
        self.act_func = nn.Sigmoid()

        """Recurrent Network architecture"""
        self.layer_1 = nn.RNNCell(in_features, 128, nonlinearity = 'relu', bias = bias) 
        self.layer_2 = nn.RNNCell(128,64, nonlinearity = 'relu', bias = bias)
        self.layer_3 = nn.RNNCell(64,32, nonlinearity = 'relu', bias = bias)
        self.layer_4 = nn.RNNCell(32,16, nonlinearity = 'relu', bias = bias)
        self.layer_5 = nn.Linear(16, 1)

    def forward(self, x):
        layer_1 = self.layer_1(x)
        layer_2 = self.layer_2(layer_1)
        layer_3 = self.layer_3(layer_2)
        layer_4 = self.layer_4(layer_3)
        y_hat = self.layer_5(layer_4)   
        return y_hat


class FCNetwork(nn.Module): #Fully Connected Neural Network
    """Fully Connected Network with Linear layers and non-linear activation."""    
    def __init__(self, in_features: int, out_features: int, bias: bool):
        
        super(FCNetwork, self).__init__()

        """activation functions"""
        self.act_func = nn.ReLU()
        #self.act_func = nn.Sigmoid()

        """Linear Network architecture"""
        self.layer_1 = nn.Linear(in_features, 128, bias = bias) 
        #self.drop_outs = nn.Dropout()
        #self.layer_2 = nn.Linear(256, 256, bias = bias)
        #self.layer_3 = nn.Linear(256, out_features, bias = bias)
        self.layer_2 = nn.Linear(128, 64, bias = bias)
        self.layer_3 = nn.Linear(64, 16, bias = bias)
        self.layer_4 = nn.Linear(16, out_features, bias = bias)
        #self.layer_5 = nn.Linear(128, out_features, bias = bias)
                  
    def forward(self, x):
        layer_1 = self.act_func(self.layer_1(x))
        layer_2 = self.act_func(self.layer_2(layer_1))
        #y_hat = self.layer_3(layer_2)
        layer_3 = self.act_func(self.layer_3(layer_2))
        y_hat = self.layer_4(layer_3)
        #y_hat = self.layer_5(layer_4)   
        return y_hat

class LinearNetwork(nn.Module): #Linear Neural Network
    """Fully Connected Network with Linear layers and non-linear activation."""    
    def __init__(self, in_features: int, out_features: int, bias: bool):
        
        super(LinearNetwork, self).__init__()
        #activation functions
        #self.act_func = nn.ReLU()
        self.act_func = nn.Sigmoid()
        #Linear Network architecture
        self.layer_1 = nn.Linear(in_features, 64, bias = bias) 
        #self.drop_outs = nn.Dropout()
        #self.layer_2 = nn.Linear(256, 256, bias = bias)
        self.layer_2 = nn.Linear(64, out_features, bias = bias)
        #self.layer_4 = nn.Linear(256, 128, bias = bias)
        #self.layer_5 = nn.Linear(128, out_features, bias = bias)
                                 
    def forward(self, x):
        layer_1 = self.layer_1(x)
        y_hat = self.layer_2(layer_1)
        #layer_3 = self.act_func(self.layer_3(layer_2))
        #layer_4 = self.act_func(self.layer_4(layer_3))
        #y_hat = self.layer_5(layer_4)   
        return y_hat