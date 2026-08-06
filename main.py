from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pypdf import PdfReader
from docx import Document
import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API_KEY_ERROR")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

# Accepted resume formats, in the order they're searched for.
SUPPORTED_RESUME_EXTENSIONS = (".pdf", ".docx")

# Optional: set this to an exact filename (e.g. "My_Resume.docx") if you
# want to force a specific file. Leave it empty to auto-detect — the app
# will use the first .pdf or .docx file it finds next to main.py.
RESUME_FILE = ""

app = FastAPI(title="Candidate Chatbot API")

# Allow the frontend (any origin) to call this API. Needed any time the
# frontend is opened from a different origin than the backend, e.g. during
# local development or if you host frontend/backend on different domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skill_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experience: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []


resume_schema = Resume.model_json_schema()


class ChatRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# The resume PDF is parsed by the LLM ONCE and cached in memory, instead of
# being re-read and re-parsed on every single chat message. This is what
# keeps each HR question fast (one LLM call instead of two) and keeps your
# Groq usage low.
# ---------------------------------------------------------------------------
_resume_cache: Resume | None = None


def read_pdf(file_path: Path) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_docx(file_path: Path) -> str:
    document = Document(file_path)
    text = ""
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text += cell.text + "\n"
    return text


def read_resume_text(file_path: Path) -> str:
    """Extract text from a resume file, dispatching by extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(file_path)
    if suffix == ".docx":
        return read_docx(file_path)
    raise HTTPException(
        status_code=500,
        detail=f"Unsupported resume file type '{suffix}'. Use .pdf or .docx.",
    )


def find_resume_file() -> Path:
    """
    Resolve which resume file to use:
    1. If RESUME_FILE is set, use it exactly (error if missing).
    2. Otherwise, auto-detect the first .pdf or .docx file next to main.py.
    """
    if RESUME_FILE:
        path = Path(RESUME_FILE)
        if not path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Resume file '{RESUME_FILE}' was not found next to main.py.",
            )
        return path

    here = Path(".")
    for ext in SUPPORTED_RESUME_EXTENSIONS:
        matches = sorted(here.glob(f"*{ext}"))
        if matches:
            return matches[0]

    raise HTTPException(
        status_code=500,
        detail="No resume file found. Add a .pdf or .docx resume next to main.py "
        "(or set RESUME_FILE to an exact filename).",
    )


def parse_resume(resume_text: str) -> Resume:
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return Resume(**data)


def get_resume() -> Resume:
    """Return the cached, parsed resume, parsing it on first use only."""
    global _resume_cache
    if _resume_cache is None:
        path = find_resume_file()
        text = read_resume_text(path)
        if not text.strip():
            raise HTTPException(
                status_code=500,
                detail=f"Could not extract any text from '{path.name}'. "
                "It may be a scanned/image-based document.",
            )
        _resume_cache = parse_resume(text)
    return _resume_cache


def ask_candidate(question: str, resume: Resume) -> str:
    system_prompt = f"""
You are an AI assistant representing a job candidate.

Below is everything you know about the candidate:

{resume.model_dump_json(indent=2)}

Rules:
1. Answer only using this information.
2. Never hallucinate.
3. If information is unavailable, say
   "I don't have enough information to answer that."
4. Be professional.
5. Answer as if HR is interviewing the candidate.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


@app.get("/api/profile")
def profile():
    """A small public summary the frontend uses to render the profile card."""
    resume = get_resume()
    return {
        "name": resume.name,
        "total_experience_years": resume.total_experience_years,
        "skills": resume.skills[:10],
        "education": resume.education,
    }


@app.post("/chat")
def chat(request: ChatRequest):
    resume = get_resume()
    answer = ask_candidate(request.question, resume)
    return {"answer": answer}


# Serve the frontend (frontend/index.html) at "/".
# IMPORTANT: this must be the LAST route registered — routes defined above
# are matched first, and anything not matched falls through to these files.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")