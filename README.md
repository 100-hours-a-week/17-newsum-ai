# NewSum AI Service

NewSum AI Service is a FastAPI-based application that generates comics based on news articles or user opinions. It utilizes a LangGraph workflow to process queries, analyze content, and generate creative scenarios and images.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Development](#development)
- [License](#license)

## Project Overview

NewSum AI Service transforms user queries or news content into engaging comic strips through a sequence of AI-powered processes. The service uses a modular architecture with LangGraph for workflow management, seamlessly integrating multiple AI services including language models, search tools, and image generation.

## Architecture

The system is built on a modular architecture with the following components:

1. **FastAPI Application**: Provides REST API endpoints for comic generation and status checking
2. **LangGraph Workflow**: Coordinates the sequential processing nodes from query analysis to image generation
3. **Service Components**:
   - LLM Service: Handles language model interactions
   - Image Service: Manages image generation via external API
   - Google Search Tool: Performs web searches for research
   - Database Client: Stores workflow status and results
   - Additional utility services (Translation, Spam detection, etc.)

## Key Features

- **Asynchronous Processing**: Non-blocking workflow execution with status updates
- **Sequential Node Processing**: Nine specialized processing nodes that handle discrete stages
- **Background Task Management**: FastAPI background tasks for managing long-running operations
- **Robust Error Handling**: Comprehensive error management and logging throughout the workflow
- **Flexible Configuration**: Environment-based configuration for all services

## Workflow

The main workflow consists of 9 sequential nodes:

1. **Initialize**: Sets up workflow state and validates query
2. **Analyze Query**: Examines the user query and sets context
3. **Understand and Plan**: Formulates a search strategy based on the query
4. **Execute Search**: Performs web searches using the Google Search Tool
5. **Report Generation**: Creates a comprehensive report from search results
6. **Save Report**: Stores the generated report for future reference
7. **Comic Ideation**: Generates creative comic ideas based on the report
8. **Scenario Generation**: Develops detailed scenarios from the selected idea
9. **Image Generation**: Creates images for each scene in the scenario

## Installation

### Prerequisites

- Python 3.8+
- FastAPI
- LangGraph
- Required environment variables (see `.env.sample`)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/17-team-4cut.git
   cd 17-team-4cut/ai
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Copy the environment template and set up your variables:
   ```bash
   cp .env.sample .env
   # Edit .env with your configuration settings
   ```

## Usage

### Running the Server

Start the FastAPI server:

```bash
python main.py
```

The server will run on http://0.0.0.0:8090 by default.

### API Endpoints

#### Generate Comic

```
POST /api/v1/comics
```

Request body example:
```json
{
  "writer_id": "writer1",
  "data": {
    "query": "Effects of climate change on coastal cities",
    "site": {
      "news": ["bbc.com", "cnn.com"],
      "research_paper": ["nature.com", "science.org"]
    }
  }
}
```

Response example:
```json
{
  "comic_id": "9d8be988-833e-4500-8e38-8d45ca150449",
  "status": "PENDING",
  "message": "만화 생성 작업이 수락되어 백그라운드에서 시작되었습니다."
}
```

#### Check Comic Status

```
GET /api/v1/comics/status/{comic_id}
```

Response example:
```json
{
  "comic_id": "9d8be988-833e-4500-8e38-8d45ca150449",
  "status": "DONE",
  "message": "워크플로우 성공적으로 완료됨.",
  "query": "Effects of climate change on coastal cities",
  "writer_id": "writer1",
  "user_site_preferences_provided": true,
  "timestamp_accepted": "2024-05-07T08:30:00.000Z",
  "timestamp_start": "2024-05-07T08:30:01.123Z",
  "timestamp_end": "2024-05-07T08:32:45.678Z",
  "duration_seconds": 164.555,
  "result": {
    "trace_id": "9d8be988-833e-4500-8e38-8d45ca150449",
    "final_stage": "n09_image_generation",
    "original_query": "Effects of climate change on coastal cities",
    "report_content_length": 8750,
    "saved_report_path": "results/reports/report_9d8be988.html",
    "comic_ideas_count": 3,
    "comic_ideas_titles": ["Rising Tides", "Coastal Defenders", "City Adaptation"],
    "selected_comic_idea_title": "Rising Tides",
    "comic_scenarios_count": 1,
    "scenario_scenes_approx": 4,
    "generated_comic_images_count": 4,
    "generated_images_summary": [
      "Scene 1: /path/to/image1.png",
      "Scene 2: /path/to/image2.png",
      "Scene 3: /path/to/image3.png",
      "Scene 4: /path/to/image4.png"
    ]
  }
}
```

## Configuration

The application uses environment variables for configuration. See `.env.sample` for examples of required variables.

Key configuration parameters:

- `IMAGE_SERVER_URL`: URL of the image generation service
- `IMAGE_SERVER_API_TOKEN`: Authentication token for the image service
- `IMAGE_STORAGE_PATH`: Local path to store generated images
- Additional LLM and service configurations

## Development

The project structure follows a modular approach:

```
ai/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints.py       # API route definitions
│   │       ├── schemas.py         # Pydantic models for requests/responses
│   │       └── background_tasks.py # Background task handling
│   ├── config/
│   │   └── settings.py            # Application settings
│   ├── nodes/
│   │   ├── n01_initialize_node.py # Workflow node 1
│   │   ├── n02_analyze_query_node.py
│   │   └── ... (nodes 3-9)
│   ├── services/
│   │   ├── database_client.py     # Database interactions
│   │   ├── image_service.py       # Image generation service
│   │   ├── llm_service.py         # Language model service
│   │   └── ... (other services)
│   ├── tools/
│   │   └── search/
│   │       └── Google_Search_tool.py # Google search functionality
│   ├── utils/
│   │   └── logger.py              # Logging configuration
│   └── workflows/
│       ├── main_workflow.py       # LangGraph workflow definition
│       └── state.py               # Workflow state structure
├── generated_images/              # Storage for generated images
├── results/                       # Storage for reports and other outputs
├── .env                           # Environment variables
├── .env.sample                    # Environment variable template
├── logging_config.yaml            # Logging configuration
└── main.py                        # Application entry point
```

## License

[Include appropriate license information here]
