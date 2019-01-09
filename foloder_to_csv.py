import os

data_dir = "./csv_deep_fashion"

dir_gen = os.walk(data_dir)




label_lst = next(dir_gen)[1]
print(label_lst)

len_label = len(label_lst)

print(",".join(label_lst))


        
# for (root, dir, filename) in dir_gen:
#     print(root, filename)
#
#     for i in range(len(filename)):
#         print(os.path.join(root, filename[i]))
#     print("\n")


with open("./folder_to_csv_deep_fashion.csv", "w") as f:
    f.write(",".join(label_lst)+"\n")

    for (root, dir, filename) in dir_gen:
        for i in range(len(filename)):
            if i+1 < len(filename):
                f.write(os.path.join(root, filename[i])+",")
            else:
                f.write(os.path.join(root, filename[i]))

        f.write("\n")

