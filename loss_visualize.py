import torch
import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(0, 300)
loss = torch.tensor(torch.load('depth_loss'))

mean_loss=[]
for i in range(300):
    ml=loss[i*100:(i+1)*100].mean()
    mean_loss.append(ml)

plt.figure()
plt.plot(epochs*100, mean_loss, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Depth_Loss')
plt.title('Training Loss')
plt.savefig('2.png')