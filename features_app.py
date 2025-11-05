# features_app.py — Writer-focused FastAPI backend for Script-Dwaar (Markdown-first)
# Updated: return JSON { result, html } for /generate/format and avoid double-bolding.
# NEW: Dialogue selection validator + friendly headers for all tools.

from __future__ import annotations

import os
import datetime
import re
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

# -------------------------------
# Database (SQLite by default)
# -------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local.db")

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    ForeignKey,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id: int = Column(Integer, primary_key=True, index=True)
    email: str = Column(String(255), unique=True, nullable=False)
    role: str = Column(String(50), nullable=False)


class Submission(Base):
    __tablename__ = "submissions"
    id: int = Column(Integer, primary_key=True, index=True)
    user_id: int = Column(Integer, nullable=False)
    prompt: str = Column(Text, nullable=False)
    generated_text: str = Column(Text, nullable=False)
    created_at: datetime.datetime = Column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )


# --- Scene management & version control ---
class Scene(Base):
    __tablename__ = "scenes"
    id: int = Column(Integer, primary_key=True, index=True)
    user_id: int = Column(Integer, nullable=False, index=True)
    title: str = Column(String(255), nullable=False)
    position: int = Column(Integer, nullable=False, default=0)
    created_at: datetime.datetime = Column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )
    updated_at: datetime.datetime = Column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )
    versions = relationship("SceneVersion", back_populates="scene", cascade="all,delete")


class SceneVersion(Base):
    __tablename__ = "scene_versions"
    id: int = Column(Integer, primary_key=True, index=True)
    scene_id: int = Column(Integer, ForeignKey("scenes.id"), nullable=False, index=True)
    version: int = Column(Integer, nullable=False)
    content: str = Column(Text, nullable=False)
    created_at: datetime.datetime = Column(
        DateTime, nullable=False, default=datetime.datetime.utcnow
    )
    scene = relationship("Scene", back_populates="versions")


# Create tables (for dev). In prod use migrations (Alembic).
os.makedirs(os.path.dirname("./local.db") or ".", exist_ok=True)
Base.metadata.create_all(bind=engine)

# -------------------------------
# Firestore (optional for dev)
# -------------------------------
USE_FIRESTORE = os.getenv("USE_FIRESTORE", "0") == "1"
firestore_client = None
if USE_FIRESTORE:
    try:
        from google.cloud import firestore  # type: ignore

        firestore_client = firestore.Client()
        print("[Firestore] Enabled")
    except Exception as exc:
        print(f"[Firestore disabled] Could not init client: {exc}")
        firestore_client = None
else:
    print("[Firestore] Disabled (set USE_FIRESTORE=1 to enable)")

# -------------------------------
# OpenAI (optional) — works with openai>=1.x / 2.x
# -------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    try:
        # Note: adapt if you use a different OpenAI SDK
        from openai import OpenAI  # type: ignore

        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("[OpenAI] Client initialized (model:", os.getenv("OPENAI_MODEL", "gpt-4o-mini"), ")")
    except Exception as exc:
        print(f"[OpenAI] Failed to init client, using stub. Reason: {exc}")
        openai_client = None
else:
    print("[OpenAI] No API key set — using stub outputs")

# -------------------------------
# FastAPI app + CORS
# -------------------------------
app = FastAPI(title="AI Writing Assistant API", version="1.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Schemas
# -------------------------------
class TextRequest(BaseModel):
    user_id: int
    prompt: str
    model_params: Optional[Dict[str, Any]] = None


class TextResponse(BaseModel):
    result: str


class OutlineRequest(BaseModel):
    user_id: int
    premise: str
    structure: Optional[str] = Field(
        "three-act", description="three-act | tv-pilot | mini-room | hero's journey"
    )
    include_beats: bool = True
    include_scene_list: bool = True
    model_params: Optional[Dict[str, Any]] = None


class FormatRequest(BaseModel):
    user_id: int
    prompt: str
    model_params: Optional[Dict[str, Any]] = None


class TranslateRequest(BaseModel):
    user_id: int
    text: str
    target_language: str = Field(..., description="Language to translate to, e.g. 'Hindi' or 'hi' or 'fr' or 'French')")
    preserve_formatting: bool = True
    model_params: Optional[Dict[str, Any]] = None


# -------------------------------
# Helpers
# -------------------------------
def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


def _stub_markdown(title: str, body_lines: List[str]) -> str:
    bullets = "\n".join(f"- {ln}" for ln in body_lines)
    return f"**{title}**\n\n{bullets}\n"


async def _stub_generator(title: str):
    # kept for compatibility if call_ai_model uses it internally
    return _stub_markdown(
        f"{title} (Stub)",
        [
            "OpenAI API key not set — returning deterministic stub.",
            "Your input was captured and would be processed similarly with the model.",
            "Set OPENAI_API_KEY to get full-quality outputs.",
        ],
    )


async def _collect_response_content(resp) -> str:
    """
    Helper to extract string content from a variety of response shapes.
    """
    try:
        if getattr(resp, "choices", None):
            first = resp.choices[0]
            if hasattr(first, "message") and getattr(first.message, "content", None) is not None:
                return first.message.content
            if isinstance(first, dict) and isinstance(first.get("message"), dict):
                return first["message"]["content"]
        return str(resp)
    except Exception:
        return str(resp)


async def call_ai_model(prompt: str, *, max_tokens: int = 2000, title: str = "Result", model_params: Optional[Dict[str, Any]] = None, stream: bool = False) -> str:
    """
    Non-streaming wrapper to generate text with OpenAI (or stub).
    Although 'stream' parameter exists for compatibility, endpoints here always call non-streaming.
    Returns a string.
    """
    mp = model_params or {}
    temperature = mp.get("temperature", 0.7)
    m_max_tokens = int(mp.get("max_tokens", max_tokens))
    model_name = mp.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # No client -> stub
    if not openai_client:
        return _stub_markdown(
            f"{title} (Stub)",
            [
                "OpenAI API key not set — returning deterministic stub.",
                "Your input was captured and would be processed similarly with the model.",
                "Set OPENAI_API_KEY to get full-quality outputs.",
            ],
        )

    # Non-streaming generation
    try:
        resp = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=m_max_tokens,
            temperature=temperature,
        )
        content = await _collect_response_content(resp)
        return content.strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI generation failed: {exc}")


def save_to_relational_db(user_id: int, prompt: str, generated_text: str) -> int:
    with SessionLocal() as session:
        submission = Submission(
            user_id=user_id,
            prompt=prompt,
            generated_text=generated_text,
        )
        session.add(submission)
        session.commit()
        session.refresh(submission)
        return submission.id


def save_to_firestore(collection: str, doc_id: str, data: dict) -> None:
    if not firestore_client:
        return  # no-op in dev
    firestore_client.collection(collection).document(doc_id).set(data)


# -------------------------------
# Typographic formatting helper
# -------------------------------
def _is_title_case(s: str) -> bool:
    words = [w for w in re.split(r"\s+", s) if w]
    if not words:
        return False
    capped = sum(1 for w in words if w[0].isupper())
    return capped >= max(1, len(words) // 2)


def _already_bolded(s: str) -> bool:
    """
    Return True if the string already contains bold markers in common places.
    This checks:
      - whole-line bold: **...**
      - leading bold tokens: **Label:** or **LABEL**
      - inline <strong> tags
    """
    if not isinstance(s, str):
        return False
    s_strip = s.strip()
    # whole-line bold
    if re.match(r'^\*\*.*\*\*$', s_strip):
        return True
    # starts with bold token like "**Label:" or "**LABEL**" or "**Label:**"
    if s_strip.startswith("**") or s_strip.startswith("__"):
        return True
    # HTML strong tag
    if "<strong>" in s_strip and "</strong>" in s_strip:
        return True
    # fallback
    return False


def apply_typographic_formatting(text: str) -> str:
    """
    Apply heuristics to bold obvious titles/headings, scene slugs (INT./EXT.), ALL-CAPS character names,
    and labels-before-colon. Returns Markdown text with **bold** markers.
    """
    if not isinstance(text, str):
        return text

    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    out_lines = []
    total_lines = len(lines)

    for i, line in enumerate(lines):
        orig_line = line
        stripped = line.strip()

        if not stripped:
            out_lines.append(line)
            continue

        # Scene heading detection
        if re.match(r"^(INT|EXT|INT/EXT|EXT/INT)\b[^\n]*", stripped, flags=re.IGNORECASE):
            if not _already_bolded(stripped):
                leading_ws = line[:len(line) - len(line.lstrip())]
                out_lines.append(f"{leading_ws}**{stripped}**")
                continue

        # Label before colon (avoid double-bolding if label already bold)
        m_label = re.match(r"^([A-Za-z0-9][A-Za-z0-9 \-]{0,80}?)\s*:\s*(.*)$", stripped)
        if m_label:
            label = m_label.group(1).strip()
            rest = m_label.group(2)
            # If label already begins with bold token, keep as-is
            if stripped.startswith("**") or stripped.startswith("__") or "<strong>" in stripped:
                out_lines.append(orig_line)
                continue
            # Only bold short labels
            if 1 <= len(label.split()) <= 6 and len(label) <= 60 and not _already_bolded(label):
                leading_ws = line[:len(line) - len(line.lstrip())]
                new = f"{leading_ws}**{label}:**"
                if rest:
                    new += f" {rest}"
                out_lines.append(new)
                continue

        # ALL-CAPS short line (likely character or slug)
        if stripped.isupper() and any(c.isalpha() for c in stripped):
            words = stripped.split()
            if len(words) <= 5 and len(stripped) <= 60 and not _already_bolded(stripped):
                leading_ws = line[:len(line) - len(line.lstrip())]
                out_lines.append(f"{leading_ws}**{stripped}**")
                continue

        # General headings heuristic
        num_words = len(stripped.split())
        short_line = len(stripped) <= 60 and num_words <= 8
        all_caps = stripped.upper() == stripped and any(c.isalpha() for c in stripped)
        ends_colon = stripped.endswith(":")
        prev_blank = (i == 0) or (lines[i - 1].strip() == "")
        next_blank = (i == total_lines - 1) or (lines[i + 1].strip() == "")
        surrounded_blank = prev_blank and next_blank

        if short_line and (all_caps or _is_title_case(stripped) or ends_colon or surrounded_blank):
            if not _already_bolded(stripped):
                leading_ws = line[:len(line) - len(line.lstrip())]
                out_lines.append(f"{leading_ws}**{stripped}**")
                continue

        # Bullet heading detection
        m_bullet = re.match(r"^(\s*[\-\*\u2022]|\s*\d+\.)\s*(.{1,80})$", line)
        if m_bullet:
            marker = m_bullet.group(1)
            rest = m_bullet.group(2).strip()
            if len(rest) <= 60 and not _already_bolded(rest) and (rest == rest.title() or rest.isupper()):
                new = f"{marker} **{rest}**"
                out_lines.append(new)
                continue

        out_lines.append(orig_line)

    formatted = "\n".join(out_lines)
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
    return formatted


# -------------------------------
# NEW: Content-type detectors & friendly wrappers
# -------------------------------
def is_probable_dialogue(text: str) -> bool:
    """
    Heuristic to detect if 'text' looks like dialogue suitable for dialogue analysis.
    Returns True if we see at least 2 dialogue-ish lines.
    Matches patterns like:
      - NAME: line
      - UPPERCASE NAME on its own line followed by a non-empty line (classic screenplay)
      - Lines starting with quotes or a dash that look like speech
    """
    if not isinstance(text, str):
        return False
    lines = [ln.rstrip() for ln in text.splitlines()]
    score = 0

    name_colon = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 .'\-]{1,40}\s*:\s+\S")
    upper_name = re.compile(r"^\s*[A-Z][A-Z0-9 .'\-]{1,40}\s*$")
    scene_slug = re.compile(r"^\s*(INT|EXT|INT/EXT|EXT/INT)\b", re.IGNORECASE)

    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        # Pattern 1: NAME: dialogue
        if name_colon.match(ln):
            score += 1
            continue
        # Pattern 2: UPPERCASE NAME line, followed by a probable speech line (not a scene slug, not empty)
        if upper_name.match(ln) and not scene_slug.match(ln):
            nxt = ""
            # find next non-empty
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j].strip():
                    nxt = lines[j].strip()
                    break
            if nxt and not scene_slug.match(nxt) and not upper_name.match(nxt):
                score += 1
                continue
        # Pattern 3: quoted or dashed speech
        if ln.strip().startswith(("\"", "“", "‘", "-", "—")) and len(ln.strip()) > 3:
            score += 0.5  # weaker signal

    return score >= 2


def with_friendly_header(header: str, body: str) -> str:
    """
    Prepend a friendly header to the body text. Keeps schemas unchanged.
    """
    header = header.strip()
    body = (body or "").lstrip()
    return f"{header}\n\n{body}" if body else header


# -------------------------------
# Core endpoints (non-streaming)
# -------------------------------
@app.post("/generate/logline")
async def generate_logline(req: TextRequest):
    ai_prompt = (
        "Format your entire response in Markdown with clear bold section titles and bullet points.\n\n"
        "**LOG LINE** — Write a compelling logline (1–2 sentences) for the concept below.\n"
        "- Emphasize protagonist, conflict, and stakes.\n\n"
        f"**Concept:** {req.prompt}\n"
    )

    result = await call_ai_model(ai_prompt, max_tokens=2000, title="Logline", model_params=req.model_params, stream=False)
    result = with_friendly_header("Here’s your logline:", result)
    # Save
    try:
        submission_id = save_to_relational_db(req.user_id, req.prompt, result)
        save_to_firestore("loglines", str(submission_id), {
            "user_id": req.user_id, "prompt": req.prompt, "result": result,
            "timestamp": _now_iso(), "type": "logline"
        })
    except Exception:
        pass
    return TextResponse(result=result)


@app.post("/generate/dialogue")
async def generate_dialogue(req: TextRequest):
    ai_prompt = (
        "Return Markdown with bold section headings and bullet points where relevant.\n\n"
        "**DIALOGUE (6–12 lines)**\n"
        "- Distinct voices & subtext, no exposition dumps.\n"
        "- Prefix lines with SPEAKER NAME.\n\n"
        f"**Scene/Beat:** {req.prompt}\n"
    )

    result = await call_ai_model(ai_prompt, max_tokens=2000, title="Dialogue", model_params=req.model_params, stream=False)
    result = with_friendly_header("Here’s your dialogue:", result)
    try:
        submission_id = save_to_relational_db(req.user_id, req.prompt, result)
        save_to_firestore("dialogues", str(submission_id), {
            "user_id": req.user_id, "prompt": req.prompt, "result": result,
            "timestamp": _now_iso(), "type": "dialogue"
        })
    except Exception:
        pass
    return TextResponse(result=result)


@app.post("/producer/evaluate")
async def evaluate_submission(req: TextRequest):
    ai_prompt = (
        "Provide a concise Markdown report with bold headings and bullet points.\n\n"
        "**PRODUCER EVALUATION**\n"
        "- **Concept & Hook**\n- **Strengths**\n- **Concerns/Risks**\n- **Marketability**\n- **Next Steps** (≤200 words total)\n\n"
        f"**Submission:** {req.prompt}\n"
    )

    result = await call_ai_model(ai_prompt, max_tokens=2000, title="Producer Evaluation", model_params=req.model_params, stream=False)
    result = with_friendly_header("Here’s your producer evaluation:", result)
    try:
        submission_id = save_to_relational_db(req.user_id, req.prompt, result)
        save_to_firestore("evaluations", str(submission_id), {
            "user_id": req.user_id, "prompt": req.prompt, "result": result,
            "timestamp": _now_iso(), "type": "evaluation"
        })
    except Exception:
        pass
    return TextResponse(result=result)


@app.post("/generate/brainstorm")
async def generate_brainstorm(req: TextRequest):
    ai_prompt = (
        "Return Markdown formatted as bold section headings with bullet lists.\n\n"
        "**BRAINSTORM — 3 OPTIONS**\n"
        "For each option include:\n"
        "- **Premise (1 sentence)**\n- **Why it helps**\n- **Twist/Obstacle**\n\n"
        f"**Context:** {req.prompt}\n"
    )

    result = await call_ai_model(ai_prompt, max_tokens=2000, title="Brainstorm", model_params=req.model_params, stream=False)
    result = with_friendly_header("Here are your brainstorm options:", result)
    try:
        submission_id = save_to_relational_db(req.user_id, req.prompt, result)
        save_to_firestore("brainstorms", str(submission_id), {
            "user_id": req.user_id, "prompt": req.prompt, "result": result,
            "timestamp": _now_iso(), "type": "brainstorm"
        })
    except Exception:
        pass
    return TextResponse(result=result)


@app.post("/generate/character")
async def generate_character(req: TextRequest):
    ai_prompt = (
        "Produce a Markdown character sheet with bold headings and bullet points.\n\n"
        "**CHARACTER SHEET**\n"
        "- **Names (3 options), Age, Archetype**\n"
        "- **Core Goal**, **Central Wound/Flaw**, **Internal vs External Stakes**\n"
        "- **Backstory (120–160 words)**\n"
        "- **Personality (6 bullets)**\n"
        "- **Relationships (3 key bonds + tensions)**\n"
        "- **Voice Guide** (word choice, pacing, humor/sarcasm)\n"
        "- **Arc Beats**: Act I / Act II Turn / Act III Resolution\n\n"
        f"**Traits Provided:** {req.prompt}\n"
    )

    result = await call_ai_model(ai_prompt, max_tokens=2000, title="Character Sheet", model_params=req.model_params, stream=False)
    result = with_friendly_header("Here’s your character sheet:", result)
    try:
        submission_id = save_to_relational_db(req.user_id, req.prompt, result)
        save_to_firestore("characters", str(submission_id), {
            "user_id": req.user_id, "prompt": req.prompt, "result": result,
            "timestamp": _now_iso(), "type": "character"
        })
    except Exception:
        pass
    return TextResponse(result=result)


@app.post("/analyze/dialogue")
async def analyze_dialogue(req: TextRequest):
    # NEW: Validate that the selection looks like dialogue; if not, return a friendly nudge.
    if not is_probable_dialogue(req.prompt or ""):
        friendly = (
            "Looks like the selection doesn’t contain clear dialogue.\n\n"
            "Tip: Please select the actual dialogue lines (e.g., `CHARACTER: line` or a character "
            "name on one line followed by what they say) and try again. 😊"
        )
        return TextResponse(result=with_friendly_header("Here’s a quick heads-up:", friendly))

    ai_prompt = f"""Return a Markdown analysis with bold headings and bullet points, plus a fenced code block for sample rewrites.

**DIALOGUE ANALYSIS**
1. **Findings** — bullets (pacing, clarity, on-the-nose, clichés)
2. **Character Voice Check** — how distinct each speaker sounds
3. **Punch-up Suggestions** — actionable bullet points
4. **Sample Rewrites (6–10 lines)** — keep meaning but improve rhythm/subtext

**Dialogue:**


{req.prompt}
"""

    result = await call_ai_model(ai_prompt, max_tokens=2000, title="Dialogue Analysis", model_params=req.model_params, stream=False)
    result = with_friendly_header("Here’s your dialogue analysis:", result)
    try:
        submission_id = save_to_relational_db(req.user_id, req.prompt, result)
        save_to_firestore("dialogue_reviews", str(submission_id), {
            "user_id": req.user_id,
            "prompt": req.prompt,
            "result": result,
            "timestamp": _now_iso(),
            "type": "dialogue_analysis"
        })
    except Exception:
        pass
    return TextResponse(result=result)


# --- Outline & structure assistance (non-streaming) ---
@app.post("/generate/outline")
async def generate_outline(req: OutlineRequest):
    ai_prompt = (
        "Create a Markdown outline with bold headings and bullet points.\n\n"
        "**OUTLINE**\n"
        "A) **Logline** (1–2 sentences)\n"
        "B) **Theme & Promise of Premise** — 3 bullets\n"
        "C) **Beat Sheet** — Three-Act numbered beats (10–15 total)\n"
        "D) **Scene List** — 8–14 scene headings with 1–2 line summaries (include only if requested)\n"
        "E) **Risks & Opportunities** — 4 bullets\n\n"
        f"**Structure:** {req.structure}\n"
        f"**Include Beats:** {req.include_beats}\n"
        f"**Include Scene List:** {req.include_scene_list}\n"
        f"**Premise:** {req.premise}\n"
    )

    result = await call_ai_model(ai_prompt, max_tokens=2000, title="Outline", model_params=req.model_params, stream=False)
    result = with_friendly_header("Here’s your outline:", result)
    try:
        submission_id = save_to_relational_db(req.user_id, req.premise, result)
        params = {}
        try:
            params = req.model_dump()
        except Exception:
            params = {"structure": req.structure, "include_beats": req.include_beats, "include_scene_list": req.include_scene_list}
        save_to_firestore("outlines", str(submission_id), {
            "user_id": req.user_id, "premise": req.premise, "params": params,
            "result": result, "timestamp": _now_iso(), "type": "outline"
        })
    except Exception:
        pass
    return TextResponse(result=result)


# --- Format / Align endpoint (NON-STREAMING, CLEAN OUTPUT) ---
@app.post("/generate/format")
async def generate_format(req: FormatRequest):
    """
    Formats a screenplay/script cleanly with:
    - No Markdown or asterisks
    - No artificial indentation
    - No extra newlines
    Returns both plain text and HTML.
    """

    # --- 1️⃣ Build clear prompt ---
    ai_prompt = (
        "Format and structure the following screenplay or script with proper screenplay formatting.\n\n"
        "Instructions:\n"
        "- Use standard screenplay conventions: scene headings (INT./EXT.), action lines, parentheticals, dialogue.\n"
        "- Capitalize scene headings (INT./EXT.) and character names.\n"
        "- Do NOT use Markdown, asterisks, or code fences.\n"
        "- Do NOT indent or add extra spacing beyond normal screenplay line breaks.\n"
        "- Remove any redundant blank lines (keep at most one blank line between sections).\n"
        "- Output should be clean, plain text with correct line spacing and capitalization.\n\n"
        "Text to format:\n\n"
        f"{req.prompt.strip()}\n"
    )

    model_params = req.model_params or {}
    max_tokens = int(model_params.get("max_tokens", 4000))

    # --- 2️⃣ Call model (non-streaming) ---
    result = await call_ai_model(
        ai_prompt,
        max_tokens=max_tokens,
        title="Format",
        model_params=model_params,
        stream=False,
    )

    cleaned = result.strip() if isinstance(result, str) else str(result)

    # --- 3️⃣ Clean raw output ---
    cleaned = re.sub(r"^```[\w-]*\n?|```$", "", cleaned)  # remove code fences
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = cleaned.replace("*", "")  # remove any stray asterisks
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()  # max 1 blank line

    # --- 4️⃣ Normalize scene headings and character names ---
    def enforce_clean_screenplay(text: str) -> str:
        lines = []
        for line in text.split("\n"):
            l = line.strip()
            if not l:
                lines.append("")
                continue

            # Scene headings (INT./EXT.)
            if re.search(r'\b(INT|EXT|INT/EXT)\b', l, flags=re.IGNORECASE):
                lines.append(l.upper())
                continue

            # Character names (short lines without punctuation)
            if (
                re.match(r'^[A-Za-z .\'-]{1,40}$', l)
                and not re.search(r'[.!?]$', l)
            ):
                lines.append(l.upper())
                continue

            # Otherwise keep line as-is
            lines.append(l)

        # Remove duplicate blank lines (max one)
        final = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        return final

    try:
        formatted_text = enforce_clean_screenplay(cleaned)
    except Exception:
        formatted_text = cleaned

    # --- 5️⃣ Apply optional typography formatting ---
    try:
        formatted_text = apply_typographic_formatting(formatted_text)
    except Exception:
        pass

    # --- 5.5️⃣ Friendly header (keeps schema; only modifies strings) ---
    friendly_header = "Here’s your formatted script:"
    formatted_text = with_friendly_header(friendly_header, formatted_text)

    # --- 6️⃣ Convert to HTML safely (no Markdown, no <p> tags) ---
    import html as _html
    html_output = _html.escape(formatted_text).replace("\n", "<br>")

    # --- 7️⃣ Save to DB and Firestore (optional) ---
    try:
        submission_id = save_to_relational_db(req.user_id, req.prompt, formatted_text)
        save_to_firestore("formats", str(submission_id), {
            "user_id": req.user_id,
            "prompt": (req.prompt or "")[:1000],
            "result": formatted_text,
            "timestamp": _now_iso(),
            "type": "format"
        })
    except Exception:
        pass

    # --- 8️⃣ Return clean plain text and HTML ---
    return JSONResponse(content={"result": formatted_text, "html": html_output})




# --- Translate endpoint (non-streaming) ---
@app.post("/generate/translate")
async def generate_translate(req: TranslateRequest):
    """
    Translates text using OpenAI (if key available) or googletrans fallback.
    Non-streaming.
    """
    def _strip_triple_fences(s: str) -> str:
        """Remove triple backticks or inline code ticks around plain text."""
        if not isinstance(s, str):
            return s
        s = re.sub(r'^\s*```[\w-]*\r?\n', '', s)
        s = re.sub(r'\r?\n\s*```\s*$', '', s)
        s = re.sub(r'^`(.*)`$', r'\1', s)
        s = re.sub(r'\n{3,}', '\n\n', s)
        return s.strip()

    target_lang = req.target_language or "Hindi"
    ai_prompt = (
        f"Translate the following text to {target_lang}. "
        f"Return only the translated text, without any code blocks or formatting fences.\n\n"
        f"Text:\n{req.text.strip()}"
    )

    model_params = req.model_params or {}
    max_tokens = int(model_params.get("max_tokens", 4000))

    if OPENAI_API_KEY:
        result_text = await call_ai_model(ai_prompt, title="Translation", max_tokens=max_tokens, model_params=req.model_params, stream=False)
        cleaned = _strip_triple_fences(result_text)
        cleaned = with_friendly_header("Here’s your translation:", cleaned)
        try:
            submission_id = save_to_relational_db(req.user_id, req.text, cleaned)
            save_to_firestore("translations", str(submission_id), {
                "user_id": req.user_id,
                "input_text": req.text,
                "translated_text": cleaned,
                "target_language": target_lang,
                "timestamp": _now_iso()
            })
        except Exception:
            pass
        return {"result": cleaned}
    else:
        try:
            from googletrans import Translator
            translator = Translator()
            translated = translator.translate(req.text, dest=target_lang[:2].lower())
            cleaned = _strip_triple_fences(translated.text)
            cleaned = with_friendly_header("Here’s your translation:", cleaned)
            return {"result": cleaned}
        except Exception:
            stub = (
                "**Translation (Stub)**\n\n"
                "- Translation backend not configured.\n"
                "- Set OPENAI_API_KEY to use the AI translator, or install googletrans to enable fallback translation.\n"
                "- Original text is returned unchanged.\n"
            )
            stub = with_friendly_header("Here’s your translation:", stub)
            return {"result": stub}


# --- Scene management & version control (kept) ---
@app.post("/scenes")
def create_scene(req: dict) -> Dict[str, Any]:
    user_id = int(req.get("user_id"))
    title = str(req.get("title") or "").strip()
    content = str(req.get("content") or "")
    position = int(req.get("position") or 0)
    if not title:
        raise HTTPException(status_code=400, detail="Title required")
    with SessionLocal() as session:
        scene = Scene(user_id=user_id, title=title, position=position)
        session.add(scene)
        session.flush()
        first_version = SceneVersion(scene_id=scene.id, version=1, content=content)
        scene.updated_at = datetime.datetime.utcnow()
        session.add(first_version)
        session.commit()
        return {"id": scene.id, "version": 1, "title": scene.title, "position": scene.position}


@app.get("/scenes", response_model=List[Dict[str, Any]] )
def list_scenes(user_id: int = Query(...)) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        scenes = (
            session.query(Scene)
            .filter(Scene.user_id == user_id)
            .order_by(Scene.position.asc(), Scene.id.asc())
            .all()
        )
        out = []
        for s in scenes:
            latest_v = (
                session.query(SceneVersion)
                .filter(SceneVersion.scene_id == s.id)
                .order_by(SceneVersion.version.desc())
                .first()
            )
            out.append(
                {
                    "id": s.id,
                    "title": s.title,
                    "position": s.position,
                    "updated_at": s.updated_at.isoformat(),
                    "latest_version": latest_v.version if latest_v else 0,
                }
            )
        return out


@app.get("/scenes/{scene_id}")
def get_scene(scene_id: int = Path(...)) -> Dict[str, Any]:
    with SessionLocal() as session:
        scene = session.query(Scene).get(scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")
        latest = (
            session.query(SceneVersion)
            .filter(SceneVersion.scene_id == scene.id)
            .order_by(SceneVersion.version.desc())
            .first()
        )
        return {
            "id": scene.id,
            "title": scene.title,
            "position": scene.position,
            "content": latest.content if latest else "",
            "version": latest.version if latest else 0,
        }


@app.put("/scenes/{scene_id}")
def update_scene(scene_id: int, req: dict) -> Dict[str, Any]:
    with SessionLocal() as session:
        scene = session.query(Scene).get(scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")
        title = req.get("title")
        content = req.get("content")
        if title is not None:
            scene.title = str(title)
        new_version = None
        if content is not None:
            last = (
                session.query(SceneVersion)
                .filter(SceneVersion.scene_id == scene.id)
                .order_by(SceneVersion.version.desc())
                .first()
            )
            next_ver = (last.version if last else 0) + 1
            new_version = SceneVersion(scene_id=scene.id, version=next_ver, content=str(content))
            session.add(new_version)
        scene.updated_at = datetime.datetime.utcnow()
        session.commit()
        return {
            "id": scene.id,
            "title": scene.title,
            "position": scene.position,
            "version": new_version.version if new_version else None,
        }


@app.post("/scenes/reorder")
def reorder_scenes(req: dict) -> Dict[str, Any]:
    user_id = int(req.get("user_id"))
    order = list(req.get("order") or [])
    if not isinstance(order, list):
        raise HTTPException(status_code=400, detail="order must be a list")
    with SessionLocal() as session:
        scenes = session.query(Scene).filter(Scene.user_id == user_id, Scene.id.in_(order)).all()
        order_index = {sid: idx for idx, sid in enumerate(order)}
        for s in scenes:
            s.position = order_index.get(s.id, s.position)
            s.updated_at = datetime.datetime.utcnow()
        session.commit()
        return {"ok": True, "order": order}


@app.get("/scenes/{scene_id}/history")
def scene_history(scene_id: int) -> Dict[str, Any]:
    with SessionLocal() as session:
        versions = (
            session.query(SceneVersion)
            .filter(SceneVersion.scene_id == scene_id)
            .order_by(SceneVersion.version.desc())
            .all()
        )
        return {
            "scene_id": scene_id,
            "versions": [
                {"version": v.version, "created_at": v.created_at.isoformat()} for v in versions
            ],
        }


@app.post("/scenes/{scene_id}/restore/{version}")
def restore_scene_version(scene_id: int, version: int) -> Dict[str, Any]:
    with SessionLocal() as session:
        target = (
            session.query(SceneVersion)
            .filter(SceneVersion.scene_id == scene_id, SceneVersion.version == version)
            .first()
        )
        if not target:
            raise HTTPException(status_code=404, detail="Version not found")
        last = (
            session.query(SceneVersion)
            .filter(SceneVersion.scene_id == scene_id)
            .order_by(SceneVersion.version.desc())
            .first()
        )
        next_ver = (last.version if last else 0) + 1
        restored = SceneVersion(scene_id=scene_id, version=next_ver, content=target.content)
        session.add(restored)
        scene = session.query(Scene).get(scene_id)
        if scene:
            scene.updated_at = datetime.datetime.utcnow()
        session.commit()
        return {"scene_id": scene_id, "restored_from": version, "new_version": next_ver}


# Health check
@app.get("/healthz")
def health_check() -> dict:
    return {"status": "ok"}


# Unified /api/chat router (only supported types)
from enum import Enum
class ChatType(str, Enum):
    brainstorm = "brainstorm"
    character = "character"
    dialogue_analysis = "dialogue_analysis"
    dialogue = "dialogue"
    logline = "logline"
    evaluate = "evaluate"
    outline = "outline"

class ChatRequest(BaseModel):
    user_id: int
    type: ChatType
    prompt: str
    model_params: Optional[Dict[str, Any]] = None

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    # map types to endpoints and forward model_params where supported
    if req.type == ChatType.brainstorm:
        return await generate_brainstorm(TextRequest(user_id=req.user_id, prompt=req.prompt, model_params=req.model_params))
    if req.type == ChatType.character:
        return await generate_character(TextRequest(user_id=req.user_id, prompt=req.prompt, model_params=req.model_params))
    if req.type == ChatType.dialogue_analysis:
        return await analyze_dialogue(TextRequest(user_id=req.user_id, prompt=req.prompt, model_params=req.model_params))
    if req.type == ChatType.dialogue:
        return await generate_dialogue(TextRequest(user_id=req.user_id, prompt=req.prompt, model_params=req.model_params))
    if req.type == ChatType.logline:
        return await generate_logline(TextRequest(user_id=req.user_id, prompt=req.prompt, model_params=req.model_params))
    if req.type == ChatType.evaluate:
        return await evaluate_submission(TextRequest(user_id=req.user_id, prompt=req.prompt, model_params=req.model_params))
    if req.type == ChatType.outline:
        payload = OutlineRequest(user_id=req.user_id, premise=req.prompt, model_params=req.model_params)
        return await generate_outline(payload)
    raise HTTPException(status_code=400, detail="Unsupported type")
