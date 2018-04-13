import random
fid = open("Create_Train.csv", "r")
li = fid.readlines()
fid.close()
random.shuffle(li)
fid = open("temp.csv", "w")
fid.writelines(li)
fid.close()
