import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.batch_norm = nn.BatchNorm1d(embed_size)  # Batch normalization
        
        
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        
        # Batch normalization
        features = self.batch_norm(features)
        
        
        return features
    

    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        
        # Defining the Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # Defining the LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        
        # Defining the output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        """
        features dims: (batch_size, embed_size)
        captions dims: (batch_size, time_steps, embed_size)
        """ 
        # must be done to avoid dimension errors
        captions = captions[:,:-1]
        
        embeddings = self.embed(captions)
        
        
        hidden = self.init_hidden(features.size(0), features.device)
        
        
        # Concatenate the features and captions
        inputs = torch.cat((features.unsqueeze(1), embeddings.float()), dim=1)
        
        # Forward pass through LSTM
        outputs, _ = self.lstm(inputs, hidden)
        
        outputs = self.fc(outputs)
        
        
        return outputs
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        with torch.no_grad():
            
            result = []
            
            hidden = self.init_hidden(inputs.size(0), inputs.device) # initiating a hidden layer
            
            for step in range(max_len): 
                
                # print(inputs.shape)
                outputs, hidden = self.lstm(inputs, hidden)
                
                outputs = self.fc(outputs)
                
                # print(outputs.shape)
                max_val, max_idx = torch.max(outputs, dim = 2) # get the corresponding integer for a token.
                
                # print(type(max_idx.cpu().numpy()[0].item()))
                result.append(max_idx.cpu().numpy()[0].item())
                
                inputs = self.embed(max_idx)
                
                
                if max_idx == 1: # it means the model generated the <end> token
                    break
                    
            return result

                
                
                