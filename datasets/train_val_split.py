import random 
f = open('img_labels.txt','r')
lines = f.readlines()
total_num = len(lines)
val_lines = random.sample(lines, int(total_num/10))
train_lines = list(set(lines)-set(val_lines))

f1 = open('vehicle_train_label.txt', 'w')
f2 = open('vehicle_val_label.txt', 'w')

for i in range(len(train_lines)):
    f1.write(train_lines[i])
f1.close()

for i in range(len(val_lines)):
    f2.write(val_lines[i])
f2.close()
