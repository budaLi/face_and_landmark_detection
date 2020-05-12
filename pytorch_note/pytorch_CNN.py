import numpy as np
import sys
sys.path.append("./")
from data import *
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms as T
import torchvision

# # array signal parameters
fc = 1e9     # carrier frequency
c = 3e8      # light speed
M = 10        # array sensor number
N = 400       # snapshot number
wavelength = c / fc  # signal wavelength
d = 0.5 * wavelength  # inter-sensor distance

# # spatial filter training parameters
doa_min = -60      # minimal DOA (degree)
grid = 1         # DOA step (degree) for generating different scenarios
GRID_NUM = 120

SNR = 10
NUM_REPEAT = 1    # number of repeated sampling with random noise带有随机噪声的重复采样次数
batch_size = 32
learning_rate = 0.001
num_epoch = 1000



train_data,train_label = generate_data(M, N, d, wavelength, SNR, doa_min, NUM_REPEAT, grid, GRID_NUM)
train_data = torch.Tensor(train_data)
train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)



# 定义网络模型(卷积自编码器)  2*10*10
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=2),  # b, 8, 7, 7
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 2, 2, stride=2, padding=2),  # b, 2, 10, 10
            nn.Tanh()  # 将输出值映射到-1~1之间
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


autoencoder = autoencoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()

for epoch in range(num_epoch):
    for data in train_data:
        img, _ = data  # img是一个b*channel*width*height的矩阵
        # img = Variable(img).cuda()
         # ===================forward=====================
        output = autoencoder(img)
        a = img.data.cpu().numpy()
        b = output.data.cpu().numpy()
        loss = criterion(output, img)
         # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
           .format(epoch+1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)  # 将decoder的输出保存成图像
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')

'''
#view_data = train_data['input'].view(-1, 2*10*10).type(torch.FloatTensor)/255.
for epoch in range(num_epoch):
    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, 2*10*10)   # batch x, shape (batch, 28*28)
        b_y = b_label.view(-1, 2*10*10)   # batch y, shape (batch, 28*28)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)'''


