import torch
import torchvision
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # Загружаем натренированную resnet152
        resnet = torchvision.models.resnet152(pretrained=True)
        # Убираем линейные слои (нам нужны только CNN)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Ресайз фичей изображения к нужным размерам
        self.pooling = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, images):

        # Извлекаем 2048 "каналов" фичей по 7X7 каждый
        out = self.resnet(images)  # (batch_size, 2048, 7, 7)
        # Изменяем размер каналов до (encoded_image_size, encoded_image_size)
        out = self.pooling(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        # Переставляем местами размерности (просто для удобства)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out


class Attention(nn.Module):

    def __init__(self, word_embeddings_dim, attention_dim, encoded_image_size):
        super(Attention, self).__init__()

        self.att_encoder = nn.Linear(2048, attention_dim)
        self.att_decoder = torch.nn.Linear(word_embeddings_dim, attention_dim)
        self.att_final = torch.nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim = 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_out, decoder_out, batch_size):
        att_encoder_computed = self.att_encoder(encoder_out)  # (batch_size, encoded_image_size**2, attention_dim)
        att_decoder_computed = self.att_decoder(decoder_out)  # (batch_size, attention_dim)
        att = self.att_final(self.relu(att_encoder_computed + att_decoder_computed.unsqueeze(1))).squeeze(2)  # (batch_size, encoded_image_size**2)
        att_weights = self.softmax(att)  # (batch_size, 2048)

        encoder_weighted = (encoder_out * att_weights.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return encoder_weighted


class Decoder(nn.Module):

    def __init__(self, vocab_size, word_embeddings_dim, attention_dim, decoder_hidden_size, encoded_image_size):
        super(Decoder, self).__init__()

        self.encoded_image_size = encoded_image_size
        self.decoder_hidden_size = decoder_hidden_size
        self.word_embeddings_dim = word_embeddings_dim
        self.vocab_size = vocab_size
        self.encoded_image_size = encoded_image_size

        self.LSTMCell = torch.nn.LSTMCell(2048 + word_embeddings_dim,
                                          hidden_size=decoder_hidden_size, bias = True)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_embeddings_dim)
        self.Attention = Attention(word_embeddings_dim, attention_dim, encoded_image_size)
        self.linear = torch.nn.Linear(decoder_hidden_size, vocab_size)

        self.h_init = torch.nn.Linear(2048, decoder_hidden_size)
        self.c_init = torch.nn.Linear(2048, decoder_hidden_size)

    def forward(self, captions, encoder_out, captions_lengths):


        # Размер батча (нужно для инициализации векторов)
        batch_size = encoder_out.size()[0]
        # Инициализирум вектор предсказаний размерности  # (batch_size, max(captions_length), vocab_size) \
        # (то есть для каждого наблюдения имеет вектор, состоящий из векторов вероятности появления каждого слова на конкретном месте предложения)
        predictions = torch.zeros(batch_size, max(captions_lengths), self.vocab_size).to(device) # (batch_size, max(captions_length), vocab_size)
        predictions[:, 0, 0] = 1 # ставим вероятность в 1 для первого слова
        # Выравниваем каналы (то есть было 2048 матриц размерностями encoded_image_size, encoded_image_size, \
        # а стало 2048 векторов размерностями encoded_image_size**2)
        encoder_out = encoder_out.view(batch_size, -1, 2048) # (batch_size, max(captions_length), 2048)
        # Сортируем наблюдения в порядке убывания длины предложения
        captions_lengths, sort_ind = captions_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind] # (batch_size, max(captions_length), 2048)
        captions = captions[sort_ind]
        # Делаем из слов эмбеддинги
        embeddings = self.embedding(captions) # (batch_size, max(captions_length), word_embeddings_dim)
        # Инициализируем вектора LSTM для первого слова (с помощью картинки)
        h = self.h_init(encoder_out.mean(dim = 1)) # (batch_size, decoder_hidden_size)
        c = self.c_init(encoder_out.mean(dim = 1)) # (batch_size, decoder_hidden_size)


        for word_n in range(1, max(captions_lengths)):
            # Количество наблюдений, для которых длина предложения больше заданной длины
            batch_size_n = sum([length > word_n for length in captions_lengths])

            # Выбираем эмбеддинг слова, стоящего на позиции word_n - 1 (то есть эмбеддинг предыдущего слова)
            decoder_out = embeddings[:, (word_n - 1)] # (batch_size, word_embeddings_dim)


            # Механизм внимания
            encoder_weighted = self.Attention(batch_size = batch_size_n,
                                              encoder_out = encoder_out[:batch_size_n],
                                              decoder_out = decoder_out[:batch_size_n]) # (batch_size, encoded_image_size**2)

            gate = self.sigmoid(self.f_beta(h[:batch_size_n]))  # gating scalar, (batch_size_t, encoder_dim)
            encoder_weighted = gate * encoder_weighted
            # Конкатенируем информцию из механизма внимания и информацию о предыдущем слове
            decoder_in = torch.cat((encoder_weighted, decoder_out[:batch_size_n]), 1) # (batch_size, encoded_image_size**2 + word_embeddings_dim)

            # Предсказываем вероятности появления слов на текущей позиции
            h, c = self.LSTMCell(decoder_in, (h[:batch_size_n], c[:batch_size_n])) # (batch_size, decoder_hidden_size)
            predictions_word = self.linear(h) # (batch_size, decoder_hidden_size)
            # Записываем информацию о предсказанных вероятностях (еще не вероятностях) в вектор
            predictions[:batch_size_n, word_n, :] = predictions_word

        return predictions, captions, captions_lengths, sort_ind