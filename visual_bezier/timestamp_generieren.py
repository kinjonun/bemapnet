import os

# 指定目标文件夹路径
folder_path = '/home/sun/Bev/BeMapNet/data/argoverse2/customer_train'

# 获取文件夹中所有文件的文件名
file_names = os.listdir(folder_path)

# 指定输出txt文件的路径
output_file = '/assets/splits/argoverse2/train_timestamp_list.txt'

# 将文件名写入到txt文件中
with open(output_file, 'w', encoding='utf-8') as f:
    for file_name in file_names:
        name_without_extension = os.path.splitext(file_name)[0]
        f.write(name_without_extension + '\n')

print(f"文件名已成功写入到 {output_file}")
