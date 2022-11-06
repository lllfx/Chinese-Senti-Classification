import torch
from tabulate import tabulate
import utils
from tqdm import tqdm
from models.text_cnn_model import TextCnnModel
from models.rnn_model import RnnModel
from models.rnn_attention_model import RnnAttentionModel
from torch.utils.data import DataLoader, Dataset
from entity import FastTextConfig

fast_text_config = FastTextConfig()
fast_text_config.vocab_path = 'ChnSentiCorp/vocab.json'
fast_text_config.train_path = 'ChnSentiCorp/train.txt'
fast_text_config.dev_path = 'ChnSentiCorp/dev.txt'

vocab, train_data, dev_data = utils.build_dataset(fast_text_config)
print('train_data', len(train_data))
print('dev_data', len(dev_data))


class TextDataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], torch.LongTensor(self.data[index][2]), torch.LongTensor(
            self.data[index][3]), torch.LongTensor(
            self.data[index][4])


BATCH_SIZE = 64
train_dataloader = DataLoader(TextDataSet(train_data), shuffle=True, batch_size=BATCH_SIZE)
dev_dataloader = DataLoader(TextDataSet(dev_data), shuffle=False, batch_size=BATCH_SIZE)

# model = TextCnnModel(len(vocab), fast_text_config.num_classes, fast_text_config.pad_size,
#                      padding_idx=len(vocab) - 1)

model = RnnAttentionModel(len(vocab), fast_text_config.num_classes, padding_idx=len(vocab) - 1)
print(model.parameters)

LR = 1e-3
EPOCHS = 20
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    print_data = []
    # 训练
    model.train()
    total_loss = 0.0
    total_batch = 0
    correct = 0
    total = 0
    for batch_idx, source in enumerate(tqdm(train_dataloader), start=1):
        optimizer.zero_grad()
        label, seq_lens, x1, x2, x3 = source
        out = model.forward(x1, seq_lens)

        loss = criterion(out, label)
        total_loss += loss.item()
        total_batch += 1
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(out.data, 1)
        total += predicted.shape[0]
        correct += (predicted == label).float().sum().item()

    accuracy = 100 * correct / total
    print_data.append([epoch, 'train', accuracy, correct, total])
    # 预测
    model.eval()
    correct = 0
    total = 0
    for batch_idx, source in enumerate(tqdm(dev_dataloader), start=1):
        label, seq_lens, x1, x2, x3 = source
        out = model.forward(x1, seq_lens)

        _, predicted = torch.max(out.data, 1)
        total += predicted.shape[0]
        correct += (predicted == label).float().sum().item()
    accuracy = 100 * correct / total
    print_data.append([epoch, 'dev', accuracy, correct, total])
    print('\n', tabulate(print_data, headers=["epoch", "train/dev", "accuracy", "correct", "total"], tablefmt="grid"))
