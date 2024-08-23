# from PIL import Image
# import os
#
# # 设置你的图片文件夹路径
# data_folder = 'data0'
#
# # 子文件夹名称列表
# subfolders = ['0', '1', '2', '3']
#
# # 存储每张图片的尺寸
# dimensions = []
#
# # 遍历每个子文件夹
# for subfolder in subfolders:
#     # 构建完整的子文件夹路径
#     folder_path = os.path.join(data_folder, subfolder)
#
#     # 检查子文件夹是否存在
#     if os.path.exists(folder_path):
#         # 假设我们只读取每个子文件夹中的第一张图片
#         for filename in os.listdir(folder_path):
#             # 检查文件是否是图片（这里以.jpg为例，你可以根据需要调整）
#             if filename.lower().endswith('.png'):
#                 # 构建图片的完整路径
#                 image_path = os.path.join(folder_path, filename)
#
#                 # 打开图片
#                 with Image.open(image_path) as img:
#                     # 获取图片尺寸
#                     width, height = img.size
#
#                     # 打印图片尺寸
#                     print(f"图片 {filename} 的尺寸是：宽 {width}, 高 {height}")
#
#                     # 将尺寸添加到列表中
#                     dimensions.append((filename, width, height))
#                 break  # 只读取第一张图片
#
# # 如果需要，可以打印所有图片的尺寸
# for dim in dimensions:
#     print(f"图片 {dim[0]} 的尺寸是：宽 {dim[1]}，高 {dim[2]}")
