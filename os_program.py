import os

#first make folder
if (not os.path.exists("data")):
    os.mkdir("data")

#create a 100 folder update the data folder
elif(not os.path.exists("data")):
    for i in range(0,100):
        os.mkdir(f"data/Day{i+1}")

#os.rename(src, dist) this function use forrename the folder

folders = os.listdir("data") 
print(type (folders))
print(len(folders))

# check what file is present in this 100 folder

for folder in folders:
    print(os.listdir(f"data/{folder}"))
    #print currect diractry
    print(os.getcwd())

