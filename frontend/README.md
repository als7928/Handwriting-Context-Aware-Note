# Frontend — Handwriting Context-Aware Note

A Vue 3 + Vite single-page application.  
Upload PDF documents, view them in the built-in viewer, and query an AI about annotated content.

## Tech Stack

| Library | Role |
|---------|------|
| Vue 3 (`<script setup>`) | UI framework |
| Vite | Bundler / dev server |
| Axios | HTTP client |
| pdfjs-dist | In-browser PDF rendering |
| marked | Markdown rendering (AI responses) |
| Bootstrap 5 | UI styling |

## Layout

```
┌─────────────────────────────────────────────────┐
│  DocumentSidebar  │      PdfViewer               │
│  (document list   │  (PDF pages + highlights)    │
│   & upload)       ├──────────────────────────────│
│                   │      ChatPanel               │
│                   │  (query input + AI answer)   │
└─────────────────────────────────────────────────┘
```

## Local Development Setup

### Prerequisites
- Node.js 20+
- Backend server running on `:8000`

### 1. Install Dependencies

```bash
npm install
```

### 2. Start Dev Server

```bash
npm run dev
# Runs at http://localhost:5173
# /api/* requests are automatically proxied to http://localhost:8000 by Vite
```

## Production Build

```bash
npm run build     # outputs static files to dist/
npm run preview   # locally preview the production build
```

## Running with Docker

Using the root `docker-compose.yml` is recommended.

```bash
# from the project root
docker compose up --build
# → http://localhost:5173
```

The frontend container runs Nginx and proxies `/api` requests to `backend:8000`.  
See [nginx.conf](nginx.conf) for the proxy configuration.

## Environment Variables

In development mode, API proxying is handled by the `server.proxy` setting in [vite.config.js](vite.config.js).  
No `.env` file is needed.

