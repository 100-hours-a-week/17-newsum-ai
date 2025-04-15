## 개요
1. 원하는 스타일의 Lora 훈련용 사진 준비 (수동)
2. Dataset_Maker로 사진에 대한 Caption 제작
3. Lora_Trainer_XL 로 만들어진 데이터셋을 활용하여 Lora 제작 (epoch 수 만큼 만들어지며, 가장 높은 숫자가 최종본)
4. Test_Lora_AB 로 원본 vs Lora적용 을 비교


## 1. 이미지 수집
- 원하는 스타일(지브리, 파스텔 톤)에 맞는 이미지를 수집
- 같은 그림체를 유지해야하며, 가능하면 하나의 피사체에 대해 다양한 각도를 제공해주는게 좋음
- **google drive의 적합한 경로**에 위치해야함 (reference의 google_drive_capture 참고)
- 이미지는 최소 20장 정도 ~ 40장이 좋음
  - 더 많으면 더 좋지만, 일관된 스타일을 대량 수집하는게 어려움


## 2. Dataset_Maker
- 이미지에 대한 설명글인 **caption**을 자동 제작
- 해당 파일의 1, 3, 4번은 필수이며 (파일 최상단 텍스트 참고)
  - 5번의 Curate your tags 부분은 '구동 명령어'를 만드는거라 굳이 없어도 되긴 함 (다중 lora 사용 시에는 필요할 거 같음)
- 예시 이미지 및 제작된 caption은 reference 참고


## 3. Lora_Trainer_XL
- 준비된 데이터셋(이미지 + caption)을 활용하여 Lora 제작
- 파일 내에 '필수 순서 가이드' 에 해당하는 내용을 바꾸고 실행하면 됨
  - 여기서 num_repeats 값은 **(준비된 이미지 수) x num_repeats == 400 쯤** 이면 된다고 함


## 4. Test_Lora_AB
- 원본 Stable Diffusion XL과 Lora가 적용된 상태를 비교하는 코드
- Setup에서 lora 파일 경로 및 이름을 맞춰주고
- (Generation은 1.5와 xl에 반응하게 되어있고)
  - Pipeline 부분이 base model과 Lora를 이어주는 부분
  - apply_lora 에서 model에 lora를 더하는 load_lora_weight 실행
  - 이 외는 gpu 사용량 체크 및 캐시 해제 등
- 아래 A/B에서 prompt를 맞춰주고 실행하면 됨