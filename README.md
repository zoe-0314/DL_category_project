# 이미지 분석 기반 의류 카테고리 자동화 분류 시스템
---
## Contents
[Introduction](#💡-Introduction)   
[Result Video](#💻-Result-Video)   
[Project Background](#📚-Project-Background)   
[Project expected effect](#🙂-Project-expected-effect)   
[Data Description](#📒-Data-Description)   
[Development Process](#⚙-Development-Process)   
[Conclusion](#📋-Conclusion)   
[Related files](#📂-Related-files)

<br><br>


## 💡 Introduction
**“ 이미지 분석 기반 의류 카테고리 자동화 분류 시스템 ”**   
- 이미지 분석 기반 의류 카테고리 자동화 분류 시스템을 구현하여 수기로 분류하는 작업을 자동화하고자 함   
- YOLOv8을 이용한 각 분류 모델 생성, Flask를 이용한 관리자 웹 구현, MySQL을 이용하여 분류된 카테고리 결과를 DB에 저장까지 할 수 있는 하나의 프로세스 구현   
![1](https://github.com/parkmy0420/DL_category_project/assets/119393455/14728653-75cb-4a38-b287-2d7ea4bd45c4)
(아이콘 출처: https://docs.ultralytics.com/, https://flask.palletsprojects.com/en/2.3.x/, https://www.mysql.com/, https://colab.research.google.com)   
<br><br>

## 💻 Result Video
https://github.com/parkmy0420/DL_category_project/assets/119393455/9719ce97-3d05-491f-8a22-f2f43bb952ed

(이미지 출처 : [29cm](https://www.29cm.co.kr/home/))

<br><br>

## 📚 Project Background
- 사람들의 패션에 대한 관심도가 상승함에 따라 온라인 쇼핑 내 패션 거래액 증가
    - 통계청 조사에 따르면 2019년 - 2021년까지 꾸준히 패션 거래액 증가
    - 2021년의 경우, 온라인 쇼핑 내 패션의 비중이 25.8% 달하고 49조 7192억원을 달성
    - (출처 : [통계청, 온라인 패션 거래액 9.2% 성장한 49조7192억원](http://www.ktnews.com/news/articleView.html?idxno=122454))
- 하지만, 아직까지 쇼핑몰 내 상품을 등록할 때 상품과 적절한 카테고리를 수기로 선택해야 함.
<br>

![2](https://github.com/parkmy0420/DL_category_project/assets/119393455/54f7d407-8eb8-4aa6-8a9b-3cb972e2d61f)   

**👉 딥러닝을 이용한 이미지 분석 기반 의류 카테고리 자동화 분류 시스템을 만들어 의류 상품 등록 시 카테고리 설정을 자동화하고자 함**   
(아이콘 출처: [flaticon](https://www.flaticon.com/))   
<br><br>

## 🙂 Project expected effect
이미지 분석 기반 의류 카테고리 자동화 분류 시스템으로 자동화가 될 경우,

1. **작업 시간 단축 및 인건비 감소**
2. **카테고리 분류 작업자 판단 기준에 따라 카테고리 속성이 다르게 분류 될 수 있음 ⇒ 통일된 카테고리 속성 분류로 인한 불편함 해소 기대**
    1. 고객의 경우, 상품을 찾는 것에 대한 불편함
    2. 관리자의 경우, 상품 관리의 불편함

<br><br>

## 📒 Data Description
- AIHub K-Fashion 이미지 데이터 셋 중 일부인 약 십이만 건 데이터 활용
  - 상의, 하의, 아우터, 원피스, 소재, 소매 기장 대분류 및 하위 세부 속성 활용
- 데이터 출처 : [AIHub K-Fashion 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=realm&dataSetSn=51)
<br><br>

## ⚙ Development Process
![3](https://github.com/parkmy0420/DL_category_project/assets/119393455/1485d975-c0e3-4df2-9086-86d2d7256a71)    
<br>

### 각 모델별 추가 전처리
![image](https://github.com/parkmy0420/DL_category_project/assets/119393455/4061d2d6-caf9-4e00-abde-a4ff4b5fd617)
<br>

### YOLOv8을 이용한 각 분류 모델 생성

- **데이터가 적었던 클래스의 경우 정확성이 조금 떨어지지만, 대체적으로 준수하게 예측되고 있음**   
- 카테고리 분류 모델 (batch size = 64, epoch = 80, imgsz = 640)   
![image](https://github.com/parkmy0420/DL_category_project/assets/119393455/a8938b84-afdd-4166-9a5b-822cc66a1974)

- 소재 분류 모델 (batch size = 64, epoch = 80, imgsz = 416)   
![image](https://github.com/parkmy0420/DL_category_project/assets/119393455/71b76a58-aa45-497c-b735-f5ed94f57b3f)

- 소매 기장 분류 모델 (batch size = 32, epoch = 100, imgsz = 640)   
![image](https://github.com/parkmy0420/DL_category_project/assets/119393455/2037d749-a2ea-4505-b70a-84a5f39761ff)

<br>

### 시스템 구현
- Flask를 이용한 웹 구현   
    - 한 상품의 복수의 이미지 업로드하여 카테고리 분류 가능   
    - 카테고리 분류 결과에서 원하는 설정으로 선택해서 저장 가능   
- MySQL을 이용하여 선택한 카테고리 분류 결과 데이터 저장   
(이미지 참조 : https://www.musinsa.com/app/goods/1856181, AIHub K-Fashion 이미지 데이터 셋 이미지)   

![image](https://github.com/parkmy0420/DL_category_project/assets/119393455/ebe240ea-877b-4ce1-a2db-c75cab0e6082)
![image](https://github.com/parkmy0420/DL_category_project/assets/119393455/d74ef708-c3c6-40f6-8423-421d1fa338e3)

<br><br>

## 📋 Conclusion
- **한계점**
    - 충분히 모델을 학습시키기에 부족한 컴퓨터 자원
    - 카테고리 항목별 개수가 불균형할 경우(데이터가 충분하지 않을 경우)
    분류 정확성이 떨어질 수 있음
    - 유행에 민감한 의류 상품의 경우, 새로운 항목 추가 및 재훈련, 학습 필요
<br>

- **개선할 점**
  - 부족했던 특정 클래스 이미지 보충하여, 모델 성능 향상 필요

<br><br>

## 📂 Related files
- 폴더별 설명
  - final_model_pt : 각 모델 생성 후 가장 성능이 좋은 weight 파일   
  - server : flask를 이용한 웹 구현 관련 코드
  - sleeve_length_code : 소매기장 전처리 관련 코드

