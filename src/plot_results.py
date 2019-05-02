import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
import os

root_path='/home/eric/data/CoRe50/log'
history_file='training_default.csv'
history_path=os.path.join(root_path,history_file)
df = pd.read_csv(str(history_path))
count=df.shape[0]
list_epoch=[i+1 for i in range(count)]
print(list_epoch)
data = {"epoch":list_epoch}

f1 = pd.DataFrame(data)
val = df['val_acc']
epoch = f1['epoch']
plt.figure()
plt.plot(epoch, val, label='val')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel("val_accuracy")
plt.show()

plt.savefig('val_accuracy.png')
plt.close() 