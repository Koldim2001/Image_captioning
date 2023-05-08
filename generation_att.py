import torch
import torchvision as tv
import os
import pickle
from torchvision import transforms as T
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

with open('vocab.pkl', 'rb') as f:
    words = pickle.load(f)

# Параметры
feature_dim = 576
lstm_dim = 1024
embed_dim = 1024
attention_dim = 2048
num_hidden = 256
num_steps= 20
dict_length=len(words)
batch_size = 100


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)   # linear layer to transform encoder's output
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)   # linear layer to transform decoder's output
        self.full_attn = nn.Linear(attention_dim, 1)
    
    def forward(self, image_features, decoder_hidden): 
      # image_features = mobilenet.features(image) # [1,576,7,7]
      # decoder_hidden = LSTM_hidden nn.LSTM(input_dim, output_dim) # [B, 1, output_dim]
      # Q, K, V  V=image_features, Q=decoder_hidden, K=image_features 
        attn1 = self.encoder_attn(image_features)          # (batch_size, num_pixels, attention_dim)
        attn2 = self.decoder_attn(decoder_hidden)       # (batch_size, attention_dim)
        attn = self.full_attn(F.relu(attn1 + attn2.unsqueeze(1)))    # (batch_size, num_pixels, 1)

        # apply softmax to calculate weights for weighted encoding based on attention
        alpha = F.softmax(attn, dim=1)                  # (batch_size, num_pixels, 1) num_pixels = 7 (width) * 7 (height)= 49
        attn_weighted_encoding = (image_features * alpha).sum(dim=1)  # (batch_size, encoder_dim)
        alpha = alpha.squeeze(2)  # (batch_size, num_pixels)
        return attn_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, dict_size, encoder_dim=2048, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = dict_size
        self.dropout = dropout
        
        
        self.embed = nn.Embedding(dict_size, embed_dim)                    # embedding layer
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True) # decoding LSTMCell
        self.fc = nn.Linear(decoder_dim, dict_size)        # linear layer to find scores over vocabulary
        
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)     # attention network
        self.dropout = nn.Dropout(p=dropout)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)   # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)    # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)   # linear layer to create a sigmoid-activated gate
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch_size, encoder_dim)
        h = self.init_h(mean_encoder_out)   # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)   # (batch_size, decoder_dim)
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lens):
        # encoder_out = mobilenet.features(image) # [1,7,7,576]
     
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1) *encoder_out.size(2)
        encoder_dim = encoder_out.size(-1)
        # flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)     # (1, 49, 576)
        num_pixels = encoder_out.size(1)
        
        # embedding
        embeddings = self.embed(encoded_captions)   # (batch_size, max_caption_length, embed_dim)

        # initialize lstm state
        h, c = self.init_hidden_state(encoder_out)      # (batch_size, decoder_dim)
        decode_lens = caption_lens.tolist() # (caption_lens - 1).tolist()

        # create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lens), self.fc.weight.size(0)).to(device)
        alphas = torch.zeros(batch_size, max(decode_lens), num_pixels).to(device)

        # decode_lens = [10,8,3], encoder_out[B(3),num_pixels (49), f_dim(576)]
        # When t = 0,1,2,3.... t == 3:
        # decode_lens = [10,8,3]

        # decode_lens = [10,8], encoder_out[B(2),num_pixels (49), f_dim(576)]
        for t in range(max(decode_lens)): # max(decode_lens) = 20

            batch_size_t = sum([l > t for l in decode_lens])

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            gate = torch.sigmoid(self.f_beta(h[:batch_size_t]))     # sigmoid gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.lstm_cell(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1), #<word_t-1, image, h, c> -> word_t, c_t
                (h[:batch_size_t], c[:batch_size_t])
            )   # (batch_size_t, decoder_dim)

            # output - [B, 1, FI + FC]
            # get the next word prediction
            preds = self.fc(self.dropout(h))    # (batch_size_t, vocab_size)

            # save the prediction and alpha for every time step
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
    
        return predictions, encoded_captions, decode_lens, alphas


def predict_att(img):

    # Проверка наличия в папке models модели attention:
    if not os.path.exists('models/model_attention.pt'):
        import gdown
        print('Загрузка модели с гугл диска:')
        #https://drive.google.com/file/d/1gGo-JGPg0umUHceySjM6oWiLKZJ7wJik/view?usp=share_link
        url = "https://drive.google.com/uc?id=1gGo-JGPg0umUHceySjM6oWiLKZJ7wJik"
        output = 'models/model_attention.pt'
        gdown.download(url, output, quiet=False)


    # Трансформация изображений:
    transform = T.Compose([T.Resize(256), 
                       T.CenterCrop(224), 
                       T.ToTensor(), 
                       T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    llm_model = torch.load('models/model_attention.pt', map_location=torch.device(device))
    model = tv.models.mobilenet_v3_small(pretrained=True)

    word_to_id = {word: id for id, word in enumerate(words)}
    llm_model.to(device)
    model.to(device)
    llm_model.eval()
    model.eval()

    vocab_size = len(words)
    downscale_model_factor = 2 ** 5 # stride 2 is happened 5 times
    # id to word mapping
    rev_word_map = {id: word for id, word in enumerate(words)}

    # read and pre-process image
    img = img.convert('RGB')
    img = transform(img)    # (3, 256, 256)

    # ==========================================
    # Feature extraction. encode the image
    encoder_out = model.features(img.unsqueeze(0).to(device))
    encoder_out = encoder_out.permute(0,2,3,1)     # (1, enc_image_size, enc_image_size, feature_dim)

    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # flatten encoded image representation
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)
    # ==========================================

    # ==========================================
    # LLM init
    prev_words = torch.tensor([[word_to_id['BOS']]], dtype=torch.long).to(device)   # (1, 1)
    seqs = prev_words   # (1, 1)
    scores = torch.zeros(1, 1).to(device)     # (1, 1)
    seqs_alpha = torch.ones(1, 1, enc_image_size, enc_image_size).to(device)  # (1, 1, enc_image_size, enc_image_size)

    # start decoding
    step = 1
    h, c = llm_model.init_hidden_state(encoder_out)
    # ==========================================


    max_steps = 30
    while True:
        # ==========================================
        # Повторяем весь код инференса из llm модели (forward)
        #
        embeddings = llm_model.embed(prev_words).squeeze(1)  # (1, embed_dim)
        attention_weighted_encoding, alpha = llm_model.attention(encoder_out, h)  # (1, encoder_dim), (1, num_pixels, 1)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)     # (1, enc_image_size, enc_image_size)
        gate = F.sigmoid(llm_model.f_beta(h))      # gating scalar, (1, encoder_dim)
        attention_weighted_encoding = gate * attention_weighted_encoding
        h, c = llm_model.lstm_cell(
            torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c)
        )   # (s, decoder_dim)
        scores = llm_model.fc(h)      # (s, vocab_size)
        #
        # ==========================================
        scores = F.log_softmax(scores, dim=1)
        scores = scores.expand_as(scores) + scores    # (1, vocab_size) 
        top_score, top_word = scores.max(dim=1)     # (1)
        next_word_inds = top_word

        # add new words to sequences, alphas
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(0)], dim=1)    # (1, step + 1)
        seqs_alpha = torch.cat(
            [seqs_alpha, alpha.unsqueeze(1)], dim=1
        )   # (1W, step + 1, enc_image_size, enc_image_size)
        if next_word_inds[0] == word_to_id['EOS']:
            break
        # break if things have been going on too long
        if step > max_steps:
            break
        prev_words = next_word_inds
        step += 1
        
    i = 0
    seq = seqs[i].tolist()

    caption = [rev_word_map[ind] for ind in seq]
    
    # Сделаем первую букву заглавной и добавим в конце ... если обрывается генерация
    caption[1] = caption[1].title()
    if caption[-1] != 'EOS':
        caption.append('...')
    if caption[-1] == 'EOS':
        caption = caption[:-1]

    return " ".join(caption[1:])

