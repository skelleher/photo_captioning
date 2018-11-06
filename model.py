import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # TODO: consider adding BatchNorm to the ResNet
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.image_embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_embed(image_features)
        return image_features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.lstm_hidden = None

        # Word embedding. It accepts an vector (of any shape) of long word token(s) and maps them to a vector of <embed_size> 
        self.word_embed = nn.Embedding(vocab_size, embed_size)

        # LSTM outputs a vector of hidden_size based on input and previous history
        # It does not predict the next token directly; we still need a fully-connected layer for that.
        # NOTE: batch_first must be set to handle a standard batch order of [batch, seq, embed]; 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True, dropout = 0.4)

#        self.dropout = nn.Dropout(0.4)
        
        # Predict probability distribution over entire vocabulary
        self.word_likelihood = nn.Linear(hidden_size, vocab_size)
        
        # initialize the weights
        self.init_weights()

        
    def init_lstm_hidden(self, n_seqs):
        ''' Initializes hidden state; do this before every training epoch, and inference '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        return (weight.new(self.num_layers, n_seqs, self.hidden_size).zero_(),
                weight.new(self.num_layers, n_seqs, self.hidden_size).zero_())

    
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.word_likelihood.bias.data.fill_(0)
        # FC weights as random uniform
        self.word_likelihood.weight.data.uniform_(-1, 1)
    
    
    def forward(self, image_features, captions):
        # image_features[] is a batch of already-embedded image vectors
        # captions[] is a batch of word tokens that need to be embedded

        # Init LSTM hidden layer to take N inputs, where N = batch size
        if (not self.lstm_hidden):
            self.lstm_hidden = self.init_lstm_hidden(captions.shape[0])
            print("*** init lstm_hidden = ", self.lstm_hidden[0].shape)

        # embed captions
        caption_embeddings = self.word_embed(captions)
        
        # concatenate image features with captions, batch-wise
        image_features = image_features.unsqueeze(1)
        embeddings = torch.cat((image_features, caption_embeddings), 1)

        # pass batch of sequences (including the image) to LSTM
        # NOTE: saving the hidden state in a class variable, and passing it back in to lstm(),
        # throws error "Trying to backward through the graph a second time, but the buffers have already been freed"
        outputs, _ = self.lstm(embeddings)
        
#        outputs = self.dropout(outputs)
        
        probabilities = self.word_likelihood(outputs)

        # HACK: trim the first element because assignment code asserts(length = caption_length)
        # this is gross for training because .contiguous() does a big memcpy
        probabilities = probabilities[:, 0:-1, :].contiguous()
        
        return probabilities
    
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        hidden = states
        next_input = inputs
        tokens = []
        for i in range(max_len):
            outputs, hidden = self.lstm(next_input, hidden)
            probabilities = self.word_likelihood(outputs)
            probabilities = probabilities.squeeze()
        
            p, token = probabilities.max(0)
            tokens.append(token.item())
            
            next_input = self.word_embed(token).unsqueeze(0).unsqueeze(0)
            
        return tokens