# Mukesh Kumar Portfolio — AI Chatbot

An AI-powered chatbot that answers questions about my professional profile, as if you were interviewing me. Upload/drop a resume next to the backend, and it becomes a live, queryable "HR interview" bot — including behavioral (STAR-format) questions.

## How it works

1. **Resume parsing** — On first request, the app finds a `.pdf` or `.docx` resume file, extracts the text, and has an LLM (via Groq) parse it into structured data: name, skills, experience, education, projects, and certifications. The result is cached in memory so parsing only happens once.
2. **Behavioral prep (optional)** — If a file with "behav" in its name (`.pdf`, `.docx`, or `.txt`) is present, its contents are used to answer behavioral/HR questions ("Tell me about a time when...") in STAR format.
3. **Chat** — Questions are answered by an LLM using only the parsed resume (and behavioral doc, if present) as context, with instructions to never invent information. Responses are streamed back token-by-token for a real-time typing effect.
4. **Frontend** — A static frontend is served directly by the FastAPI backend.

## Tech stack

- **Backend:** FastAPI, Uvicorn
- **LLM:** Groq API (`openai/gpt-oss-120b`)
- **Document parsing:** `pypdf`, `python-docx`, `docx2txt`
- **Validation:** Pydantic
- **Frontend:** Static HTML/JS served from `/frontend`

## Getting started

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/)

### Setup

```bash
git clone https://github.com/mukeshmk2222/Mukesh-Kumar-Portfolio.git
cd Mukesh-Kumar-Portfolio
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Place a resume (`.pdf` or `.docx`) in the project root next to `main.py`. It's auto-detected — a filename containing "resume" or "cv" is preferred if there are multiple candidates. Optionally add a behavioral prep document (any filename containing "behav") for STAR-format answers.

### Run

```bash
uvicorn main:app --reload
```

Then open `http://localhost:8000` in your browser.

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend |
| `/api/profile` | GET | Public summary: name, years of experience, top skills, education |
| `/chat` | POST | Send `{"question": "..."}`, receive a streamed text answer |

## Project structure

```
.
├── main.py                                              # FastAPI backend
├── frontend/                                             # Static frontend (served at "/")
├── requirements.txt
├── Mukesh_Kumar_Resume.docx                               # Source resume (auto-detected)
└── HR_Behavioural_Interview_Answer_Bank_Chatbot_document.docx  # Behavioral prep material
```

## Notes

- All answers are grounded strictly in the resume/behavioral content provided — the model is instructed not to hallucinate and to say "I don't have enough information to answer that" when data is missing.
- The `/api/debug/resume` endpoint exposes raw extracted text and parsed resume data — useful for debugging extraction issues, but should be removed or protected before any public/production deployment.
