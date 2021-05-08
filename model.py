import torch
import torchvision
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):

    def __init__(self, encoded_image_size=14):
        '''
        :param encoded_image_size: each encoded channel size of image will be encoded_image_size X encoded_image_size
        '''

        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        # Load pretrained resnet model
        resnet = torchvision.models.resnet152(pretrained=True)
        # Delete FC layers and leave only CNN
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Resize CNN features from resnet to appropriate size
        self.pooling = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # Resnet parameters will not be modified during training
        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, images):

        # Obtain 2048 channels of features each of size 7X7
        out = self.resnet(images)  # (batch_size, 2048, 7, 7)
        # Reseize size to (encoded_image_size, encoded_image_size)
        out = self.pooling(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        # Change dimension places (just for convinience)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out


class Attention(nn.Module):

    def __init__(self, word_embeddings_dim, attention_dim):
        '''
        :param word_embeddings_dim: length of word embedding
        :param attention_dim: length of attention vector
        '''
        super(Attention, self).__init__()

        # Attention layer for encoder
        self.att_encoder = nn.Linear(2048, attention_dim)
        # Attention layer for decoder
        self.att_decoder = torch.nn.Linear(word_embeddings_dim, attention_dim)
        # Final layer of attention
        self.att_final = torch.nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim = 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_out, decoder_out):
        '''
        :param encoder_out: embedding of image
        :param decoder_out: embedding of previous word
        :return: weighted encoded image
        '''

        # Attention vector for image
        att_encoder_computed = self.att_encoder(encoder_out)
        # Attention vector for previous word
        att_decoder_computed = self.att_decoder(decoder_out)
        # Combining 2 attentions
        att = self.att_final(self.relu(att_encoder_computed + att_decoder_computed.unsqueeze(1))).squeeze(2)
        # Weighting image parts based on attention
        att_weights = self.softmax(att)
        encoder_weighted = (encoder_out * att_weights.unsqueeze(2)).sum(dim=1)

        return encoder_weighted


class Decoder(nn.Module):

    def __init__(self, vocab_size, word_embeddings_dim, attention_dim, decoder_hidden_size, encoded_image_size):
        '''
        :param vocab_size: number of words in corpus
        :param word_embeddings_dim: length of word embedding
        :param attention_dim: length of attention vector
        :param decoder_hidden_size: hidden size of lstm
        :param encoded_image_size: size of each encoded image channel
        '''
        super(Decoder, self).__init__()

        self.encoded_image_size = encoded_image_size
        self.decoder_hidden_size = decoder_hidden_size
        self.word_embeddings_dim = word_embeddings_dim
        self.vocab_size = vocab_size
        self.encoded_image_size = encoded_image_size
        self.LSTMCell = torch.nn.LSTMCell(2048 + word_embeddings_dim,
                                          hidden_size=decoder_hidden_size, bias = True)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_embeddings_dim)
        self.Attention = Attention(word_embeddings_dim, attention_dim)
        self.linear = torch.nn.Linear(decoder_hidden_size, vocab_size)
        self.h_init = torch.nn.Linear(2048, decoder_hidden_size)
        self.c_init = torch.nn.Linear(2048, decoder_hidden_size)
        self.f_beta = nn.Linear(decoder_hidden_size, 2048)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()



    def forward(self, captions, encoder_out, captions_lengths):
        '''
        :param captions: captions for images
        :param encoder_out: encoded images
        :param captions_lengths: lengths of captions
        :return:
        '''

        # Initialising vectors of predictions
        batch_size = encoder_out.size()[0]
        predictions = torch.zeros(batch_size, max(captions_lengths), self.vocab_size).to(device) # (batch_size, max(captions_length), vocab_size)
        # First word of each caption guruanteed to be <start>
        predictions[:, 0, 0] = 1
        # Falttening channels
        encoder_out = encoder_out.view(batch_size, -1, 2048)
        # Sort captions by their length (for faster loop)
        captions_lengths, sort_ind = captions_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        captions = captions[sort_ind]
        # Embedding each word of captions
        embeddings = self.embedding(captions)
        # Initialising lstm vectors for first word
        h = self.h_init(encoder_out.mean(dim = 1)) # (batch_size, decoder_hidden_size)
        c = self.c_init(encoder_out.mean(dim = 1)) # (batch_size, decoder_hidden_size)


        for word_n in range(1, max(captions_lengths)):
            # Number of captions with greater length
            batch_size_n = sum([length > word_n for length in captions_lengths])

            # Obtain embedding of previous word
            decoder_out = embeddings[:, (word_n - 1)] # (batch_size, word_embeddings_dim)


            # Attention mechanism
            encoder_weighted = self.Attention(encoder_out = encoder_out[:batch_size_n],
                                              decoder_out = decoder_out[:batch_size_n])

            gate = self.sigmoid(self.f_beta(h[:batch_size_n]))
            encoder_weighted = gate * encoder_weighted

            # Concatenating attention and previous word
            decoder_in = torch.cat((encoder_weighted, decoder_out[:batch_size_n]), 1)

            # Obtaining probabilities (not exectaly, because no softmax on this step) of word appearing on this step
            h, c = self.LSTMCell(decoder_in, (h[:batch_size_n], c[:batch_size_n]))
            predictions_word = self.linear(h)
            # Store probabilities (not exectaly, because no softmax on this step) in vector
            predictions[:batch_size_n, word_n, :] = predictions_word

        return predictions, captions, captions_lengths, sort_ind