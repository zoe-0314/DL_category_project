1. convert_category_labeling.ipynb 파일로 yolo 형식의 라벨로 변환 작업 수행<br>
＊ AI 허브에서 다운 받은 원본 image와 label(json 파일)은 각각 폴더명 image 및 fa에 저장<br>
＊ convert 실행하면 해당 라벨이 convert_label_yolo 파일 밑에 변환되어<br>
   yolo 형식의 label로 저장됨(yolo는 txt 파일을 label로 사용함)<br><br>

2. 이 과정은 생략해도 되는 과정이지만 label이 제대로 yolo 형식으로 변환되었는지 확인하려면 visual.ipynb<br>
   실행하여 이미지에 박스가 잘 그려지는지 확인해보면 됨(몇가지 케이스 확인결과 잘그려짐을 사전 확인함)<br><br>

3. data_split.ipynb를 실행하면 config.yaml 파일에 기재된 경로로 train, valid의 image 및 label을 8:2로 분배하여     
   복사해줌<br>
   ＊ 혹시 이동이 안된다면 경로지정이 잘못되었으므로 절대경로로 지정해줄 것<br><br>

4. 위 작업이 모두 마무리 되었다면 yolo 모델 학습할 준비가 완료됨 train.ipynb 파일을 실행하여 import 하고 train 코드 
   실행<br>
   ＊ 원하는 파라미터는 yolo doc 웹페이지 참고하며 찾아볼 것(테스크 자체가 쉬우므로 default 설정해도 무난할 것임)<br><br>

5. 학습이 제대로 되었다면 runs라는 폴더가 생성되며 해당 폴더에 들어가보면 train 실행 시킨 순서대로 폴더가 있을거임<br>
   ＊ 해당 폴더에 들어가보면 학습이 잘되었는지 확인할 수 있는 이미지 파일들이랑 학습 로그가 남아 있음<br><br>
