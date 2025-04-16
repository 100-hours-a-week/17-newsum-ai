### Ver 1. Base
flowchart TB
    %% 방향: 위->아래(Top->Bottom)

    User((User))
    Route53[Route53 DNS]

    subgraph "AWS CloudFront + S3"
        FE[React App]
    end

    subgraph "AWS EC2"
        direction TB
        Nginx[Nginx] --> Spring[Spring Boot]
        Spring --> MySQL[(MySQL: RDB)]
        Spring --> AI["AI Service\n(LangGraph + 모델)"]
        AI --> VectorDB[(Vector DB)]
        AI --> Scraper[Scraper Module]
    end

    User -->|HTTPS| Route53
    Route53 -->|HTTPS| FE
    FE -->|/api 요청| Nginx


### Ver 2. Micro
flowchart TB
    %% 프론트엔드
    subgraph "Frontend (React App - S3/CloudFront)"
        User((User))
        Admin((Admin))
        ReactApp[React App]
        User -->|View Comics| ReactApp
        Admin -->|Create Comic| ReactApp
    end

    APIGW[Nginx / API Gateway]

    %% 백엔드
    subgraph "Backend (Spring Boot)"
        Spring[Spring Boot API Server]
        RDB[(MySQL Database)]
        Spring --> RDB
    end

    %% AI 마이크로서비스 클러스터
    subgraph "AI Services Cluster"
        LangGraph["LangGraph / MCP\n(Flow Engine)"]
        LLM["LLM\n(Text Processor: 요약/시나리오)"]
        SDXL["Image Generator\n(SDXL + LoRA)"]
        VectorDB[(Vector DB)]
        Scraper["Scraper\n(뉴스 크롤링 모듈)"]

        LangGraph --> LLM
        LangGraph --> SDXL
        LangGraph --> VectorDB
        LangGraph -->|on-demand| Scraper
    end

    %% 흐름 연결
    ReactApp --> APIGW
    APIGW --> Spring
    Spring --> LangGraph
    LangGraph --> Spring
    Spring --> ReactApp


### Ver 2.1. Micro with Queue for task


### Ver 3. Local Cluster (Outer GPU)
flowchart TB
    subgraph "Frontend (CloudFront + S3)"
        User((User))
        ReactApp[React App]
        User --> ReactApp
    end

    subgraph "Backend (Spring Boot)"
        Spring[Spring Boot API Server]
        RDB[(MySQL)]
        Spring --> RDB
    end

    subgraph "AI 외부 추론 환경"
        ExternalAI["Colab / Local GPU\n(SDXL + LoRA 모델 실행)"]
        LangGraph["LangGraph (로컬 실행 워크플로우)"]
        ExternalAI -->|이미지 생성| LocalFile[로컬 이미지 파일]
        LocalFile -->|업로드| S3Storage[S3 Bucket]
    end

    ReactApp -->|API 호출| Spring
    Spring -->|만화 생성 요청 저장| RDB
    Spring -->|기사 or 시나리오 추출| LangGraph
    LangGraph --> ExternalAI
    S3Storage --> Spring
    Spring --> ReactApp
