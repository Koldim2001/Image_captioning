import torch
import torchvision as tv
import os
import pickle
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

with open('vocab.pkl', 'rb') as f:
    words = pickle.load(f)

# Параметры
feature_dim = 512 #ResNet измененный мной
num_hidden = 300
num_steps = 20
dict_length = len(words)
batch_size = 100
num_layers = 3

# Загрузка архитектуры
class LLMModel(torch.nn.Module):
    def __init__(self, dict_size, input_dim, feature_dim, output_keep_prob, num_layers, num_hidden):
        super().__init__()
        self.embed = torch.nn.Embedding(dict_size, feature_dim)
        self.feature_dim = feature_dim
        self.lstm_cell = torch.nn.LSTM(feature_dim,
                                    batch_first=True,
                                    hidden_size=num_hidden, 
                                    num_layers=num_layers, 
                                    dropout=output_keep_prob)
        self.linear = torch.nn.Linear(num_hidden, dict_size)
    def forward(self, x, feature):
        '''
        x - описание картинки [batch_size, max_len_text], где max_len_text - длина максимальной последовательности (num_steps)
        feature - фичи после обработки CNN [batch_size, feature_dim]
        '''
        x = self.embed(x)
        # Получу [batch_size, max_len_text, feature_dim] - {word indices представление}

        # feature.unsqueeze(1) #Input: [batch_size, feature_dim], Output: [batch_size, 1, feature_dim]
        x = torch.cat([feature.unsqueeze(1), x], dim=1)[:,:-1,:] 
        # Мы сконкатиноровали: Input: [batch_size, 1, feature_dim] {image vector}, [batch_size, T, feature_dim] {word indices} 
        # Output: [batch_size, 1+max_len_text-1, feature_dim]


        o, _ = self.lstm_cell(x) # выход [batch_size, 1+max_len_text-1, num_hidden]
        return self.linear(o) #[BOS, 14, 25, 87, 34, EOS, PAD, PAD...] ideal case #[batch_size, max_len_text+1, dict_size]





def predict(img, show=False):
    # show == True => выводим изображение и делаем title
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Загрузка моделей
    llm_model = torch.load('models/model.pt', map_location=torch.device(device))
    model = tv.models.resnet34(pretrained=True)
    model.fc = torch.nn.Identity()  # заменяем полносвязный слой на слой-тождественность


    # Создадим трансформации к изображениям:
    transform = T.Compose([T.Resize(256), 
                        T.CenterCrop(224), 
                        T.ToTensor(), 
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    word_to_id = {word: id for id, word in enumerate(words)}
    llm_model.to(device)
    model.to(device)

    def generate_new(img, show):
        llm_model.eval()
        model.eval()

        # получение индекса по слову
        rev_word_map = {id: word for id, word in enumerate(words)}

        # считывание и трансформация 
        img = img.convert('RGB')
        img_saved = img
        img = transform(img)    # (3, 224, 224)

        # Получим фичи из изображения
        encoder_image = model(img.unsqueeze(0).to(device))

        # LLM init
        # Инициализировал LSTM подав изображение
        step = 1
        h, c = llm_model.lstm_cell(encoder_image)
        # в с хранится текущее состояние рекуррентной LSTM ячейки 
        # (будет перезаписываться при прогоне)

        #Зададим первого слово BOS для первичного прогона:
        prev_words = torch.tensor([[word_to_id['BOS']]], dtype=torch.long).to(device)  
        seqs = prev_words   # размерность (1, 1)

        max_steps = 30  # Максимальный размер генерации если не будет EOS
        # цикл генерации
        while True:
            # Повторяем весь код инференса из llm модели (forward)
            embeddings = llm_model.embed(prev_words).squeeze(1)  # (1, embed_dim)
            # Мы текущее слово преставили в виде набора embed_dim чисел с 
            # помощью обученного слоя эмбеддинга

            # Прогон слова через LSTM с состоянием с
            h, c = llm_model.lstm_cell(embeddings, c)
            # Мы перезаписали на текущий момент состояние памяти с и получили output h
            # h имеет размер - [1, 256]

            # Прогон через линейный слой
            scores = llm_model.linear(h) # размерность [1, 11683]
            
            # С помошью greedy алгоритма берем самый вероятный предикт
            next_word_inds = torch.argmax(scores[0],dim=-1).unsqueeze(0)

            # Добавляем новое слово к уже сочиненным
            seqs = torch.cat([seqs, next_word_inds.unsqueeze(0)], dim=1)  # (1, step + 1)

            # Проверка на конец EOS для досрочного конца цикла:
            if next_word_inds[0] == word_to_id['EOS']:
                break

            # Выход по превышению лимита генерации
            if step > max_steps:
                break

            # Заменим текцих новый ответ на предыдущий для реализации новой генерации
            prev_words = next_word_inds
            step += 1

        # Превращаем сгенерированную последовательность в текст    
        seq = seqs[0].tolist()
        caption = [rev_word_map[ind] for ind in seq]

        if show:
            # Выводим изображение и как подпись результат генерации:
            plt.title(f'Prediction: {" ".join(caption)}')
            plt.imshow(img_saved)

        # Сделаем первую букву заглавной и добавим в конце ... если обрывается генерация
        caption[1] = caption[1].title()
        if caption[-1] != 'EOS':
            caption.append('...')
        return " ".join(caption[1:])

    caption = generate_new(img, show)
    return caption


