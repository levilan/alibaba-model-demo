# Alibaba Cloud AI Model Testing Platform

This is an AI model testing platform built with FastAPI (Python) and Vanilla JavaScript/HTML.

## Project Structure

- `app.py`: FastAPI backend main application (API endpoints, model registry, routing, stream handling).
- `templates/index.html`: Frontend main page (UI layout, functional tabs, login overlay).
- `static/js/app.js`: Frontend core logic (API calls, state management, UI interactions, dynamic rendering).
- `static/css/style.css`: Global styles.

## Core Modules

1. **Login Mechanism**:
   Authenticates via API Key. The frontend calls `/login`, and upon success fetches the model list from `/api/models`, hides the login overlay, and reveals `mainApp`.
2. **Text Generation**: Supports the Qwen series with SSE streaming output.
3. **Image & Video Generation**: Supports asynchronous tasks (submit task -> poll `task_id`).
4. **MuleAI**:
   A dedicated tab supporting an additional API Key, toggling between text and image generation (e.g., bound to `wan2.7-i2v-spicy`).

## Troubleshooting Guide

### 1. Blank Screen or Stuck at Login
- **Broken HTML Structure**: If `</div>` tags in `index.html` are mismatched, the `<div id="mainApp">` container may close prematurely.
- **JS Execution Halted**: If `populateSelectors()` in `app.js` cannot find a referenced `<select id="...">` element (like `imageModel` or `muleaiModel`), it will throw a TypeError and halt UI rendering.

### 2. How to Add a Model
- Add a new dictionary entry to the `MODELS` registry inside `app.py`.
- Restart the Docker container (`docker-compose restart`). The frontend will automatically populate the new dropdowns on reload.
