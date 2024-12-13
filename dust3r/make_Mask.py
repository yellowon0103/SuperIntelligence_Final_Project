import cv2
import numpy as np

num = 3

# 이미지 불러오기
original_image_path = f'/home/asc/PycharmProjects/SuperIntelligenc_Project/Resized_Image_{num}.png'  # 원본 이미지 경로
mask_image_path = f'/home/asc/PycharmProjects/SuperIntelligenc_Project/Resized_Mask_{num}.png'       # 마스크 이미지 경로

# 이미지를 읽어옵니다
original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

# 이미지 크기 확인
if original_image.shape[:2] != mask_image.shape[:2]:
    raise ValueError("원본 이미지와 마스크 이미지의 크기가 일치해야 합니다.")

# 마스크의 검정색 부분을 찾습니다 (픽셀 값이 0인 부분)
black_mask = mask_image == 0

# 원본 이미지의 동일한 위치에 검정색을 칠합니다
output_image = original_image.copy()
output_image[black_mask] = [0, 0, 0]  # 검정색으로 칠하기

# 결과 이미지 저장
output_image_path = f'/home/asc/PycharmProjects/SuperIntelligenc_Project/masked_{num}.png'
cv2.imwrite(output_image_path, output_image)
print(f"결과 이미지가 {output_image_path}에 저장되었습니다.")


