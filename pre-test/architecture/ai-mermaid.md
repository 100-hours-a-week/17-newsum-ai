### 간단 버전
flowchart TB
    %% 방향: 위->아래(Top->Bottom)

    subgraph "AWS EC2"
        direction TB
        Spring[Spring Boot]
        AI["AI LangGraph"]
        Spring <--> AI
    end

    AI --> LLM["Colab (LLM)"]
    AI --> SD["Colab (SD)"]

    SD -->|Upload Image| S3[(S3 Bucket)]
    Spring -->|Download Image| S3



### 상세 설명 버전
flowchart TD
    %% 구조적인 정보 흐름 표현

    subgraph AWS_EC2["AWS EC2"]
        Spring[Spring Boot]
        AI["AI LangGraph"]
    end

    subgraph Colab_Cloud["Google Colab"]
        LLM["Colab LLM (요약 및 title 생성)"]
        SD["Colab SD (이미지 생성)"]
    end

    subgraph Storage["AWS S3"]
        S3["S3 Bucket - 이미지 저장"]
    end

    %% 주요 흐름
    Spring -->|기사 URL 등 요청 수신| AI
    AI -->|title, URL 등 메타데이터 반환| Spring
    AI -->|기사 요약 및 시나리오 요청| LLM
    LLM -->|title, desc, 시나리오 반환| AI
    AI -->|프롬프트로 이미지 요청| SD
    SD --> |이미지 URL 반환| AI
    SD -->|이미지 업로드| S3
    Spring -->|이미지 URL 요청| S3
    S3 -->|이미지 URL 또는 파일 반환| Spring
