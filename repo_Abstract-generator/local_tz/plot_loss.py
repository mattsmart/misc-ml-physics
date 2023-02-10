import numpy as np 
import matplotlib.pyplot as plt 

with open('sgd_loss.txt') as file:
	lines = file.readlines()
	losses = [line.rstrip().split()[4] for line in lines]

losses = np.array(losses).astype(float)
validation_loss = losses[7::8]
training_loss = np.delete(losses, np.isin(losses, validation_loss))

print(validation_loss)

xval = np.arange(6, len(validation_loss)*7, 7)/7
xtrain = np.arange(len(training_loss))/7

plt.plot(xtrain, training_loss, label='training')
plt.plot(xval, validation_loss, label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()