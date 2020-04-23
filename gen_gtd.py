import os

base_path = '/home/zjkj/working_zjw/DCL_NET_II/DCL-master/datasets/CUB_200_2011/data'

train_txt = open('./train.txt', 'w')
test_txt = open('./test.txt', 'w')

txt = open('/home/zjkj/working_zjw/nts_vehicle_ii2d/img_labels.txt', 'r')
line_list = [x for x in txt.readlines()]

def sub(string, p, c):
        new = []
        for s in string:
            new.append(s)
        new[p] = c
        return ''.join(new)

dict_gt = {}
for i, line in enumerate(line_list):
	id = line.split(' ')[-1]
	#id = int(id)
	index = line.rfind(' ', 1)
	print(index)
	line = sub(line, index, '*')
	dict_gt.setdefault(id,[]).append(line)

for key in dict_gt.keys():
	len_list = len(dict_gt[key])
	for i, line in enumerate(dict_gt[key]):
		if i < int(len_list * 0.8):
			train_txt.write(line)
		if i > int(len_list * 0.9):
			test_txt.write(line)
		else:
			continue