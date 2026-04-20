![Lint Status](https://github.com/swe-students-spring2026/4-containers-team_not_found-1/actions/workflows/lint.yml/badge.svg)
![ML Client CI](https://github.com/swe-students-spring2026/4-containers-team_not_found-1/actions/workflows/ml-client.yml/badge.svg)
![Web App CI](https://github.com/swe-students-spring2026/4-containers-team_not_found-1/actions/workflows/web-app.yml/badge.svg)
![Event Logger CI](https://github.com/swe-students-spring2026/4-containers-team_not_found-1/actions/workflows/event-logger.yml/badge.svg)

# Doodle Not Found (Team: team_not_found)

A containerized machine learning application composed of three distinct subsystems designed to automatically recognize and classify user-drawn doodles over a web interface. User can use this web app to practice their doodle drawing skills.

## Team Members
* [ZhiHui Chen](https://github.com/Zhi-Hui-C)
* [Yash Pazhianur](https://github.com/yashpaz123)
* [Jeffrey Solano](https://github.com/jeffnoso)

## System Architecture

The system consists of three interconnected services:

1. **Web App**: A user-facing Flask web dashboard where clients can draw doodles to be analyzed.
2. **Machine Learning Client**: A backend Python service utilizing a pre-trained Hugging Face Quick Draw model to predict what the user drew.
3. **Database**: A centralized MongoDB instance persisting prediction metadata, drawings, and history.

## How it works
1. A user draws a doodle on the web interface
2. The drawing is sent to the Machine Learning Client
3. The Machine Learning model processes the image and predicts a label
4. The prediction and image data are stored in MongoDB
5. The Web App retrieves and displays results in the History tab

## Configuration & Usage

### 1. Start Services (Docker)
All pieces of the project are systematically orchestrated using `docker-compose`. There is no need to make any `.env` files—all default configurations are already baked into the `docker-compose.yml` and Dockerfiles!

Ensure you have Docker Desktop running, then execute:
```bash
docker-compose up --build
```
This single startup command securely initializes:
- A `mongodb` container storing the history natively mapped to port `27017`
- The `ml-client` API mounted to port `8000`
- The `web-app` proxy serving the site at port `5000`

### 2. Start Drawing
Once the instances are running, open your web browser to view the application:
[http://localhost:5001](http://localhost:5001)

Draw a doodle representing the instruction, hit "Upload," and view the predictions and original images right inside the "History" tab!
