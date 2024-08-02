import cv2
import os
from tqdm import tqdm

def create_video_from_images(image_folder, output_video_path, frame_rate=30):
    # 定义允许的图像后缀
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG"]
    
    # 获取图像文件列表
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # 排序，确保按正确的顺序读取图像
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # 读取第一张图像以获取视频尺寸
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以选择其他编码方式，如 'XVID'
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # 逐帧写入视频
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # 释放资源
    video_writer.release()
    print(f"Video saved at {output_video_path}")

