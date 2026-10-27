"""
Resume Matcher - Streamlit app using a free LLM through OpenRouter

Run on your laptop:   streamlit run app.py
The API key is read from .streamlit/secrets.toml (local) or the app's Secrets (Streamlit Cloud):
    OPENROUTER_API_KEY = "sk-or-..."
Never put the key in this file or anywhere else on GitHub.
"""
import json
import os
import re
import time

import openai
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="wide")

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"

# ------------------------------------------------------------------ the same functions as the notebook
SYSTEM_PROMPT = ("You are an experienced technical recruiter. You judge resumes fairly and only use facts "
                 "written in the resume. You never consider a candidate's name, gender, age or location.")


def build_prompt(resume_text, job_text):
    return f'''Compare the RESUME with the JOB DESCRIPTION.

Score the match from 0 to 100 using this rubric:
- Must-have skills in the job: up to 50 points
- Relevant experience (years and type of work): up to 25 points
- Nice-to-have skills: up to 15 points
- Education and projects: up to 10 points

Rules:
- Only count a skill as matched if the resume clearly shows it.
- Keep each list item short (2-6 words).

Reply with JSON only, no other text, in exactly this format:
{{
  "match_score": <number 0-100>,
  "verdict": "<Strong match | Partial match | Weak match>",
  "matched_skills": ["..."],
  "missing_skills": ["..."],
  "strengths": ["...", "..."],
  "suggestions": ["...", "...", "..."],
  "summary": "<one or two sentences>"
}}

JOB DESCRIPTION:
<job>
{job_text}
</job>

RESUME:
<resume>
{resume_text}
</resume>'''


def redact_contacts(text):
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email removed]", text)
    text = re.sub(r"\+?\d(?:[\s-]?\d){9,12}", "[phone removed]", text)
    return text


def extract_json(reply):
    reply = re.sub(r"```(?:json)?", "", reply or "")
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in the reply")
    return json.loads(reply[start:end + 1])


def clean_result(data):
    def as_list(value):
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [str(value).strip()] if value else []

    try:
        score = int(round(float(data.get("match_score", 0))))
    except (TypeError, ValueError):
        score = 0
    return {
        "match_score": max(0, min(100, score)),
        "verdict": str(data.get("verdict", "")).strip(),
        "matched_skills": as_list(data.get("matched_skills")),
        "missing_skills": as_list(data.get("missing_skills")),
        "strengths": as_list(data.get("strengths")),
        "suggestions": as_list(data.get("suggestions")),
        "summary": str(data.get("summary", "")).strip(),
    }


STOPWORDS = {"and", "with", "the", "for", "of", "in", "experience", "skills", "strong", "basic", "using"}


def unsupported_skills(result, resume_text):
    text = resume_text.lower()
    flagged = []
    for skill in result["matched_skills"]:
        words = [w for w in re.findall(r"[a-z0-9+#.]+", skill.lower()) if len(w) > 2 and w not in STOPWORDS]
        if words and not any(w in text for w in words):
            flagged.append(skill)
    return flagged


# ------------------------------------------------------------------ API key and client
def get_configured_key():
    # A key from Streamlit Secrets (secrets.toml locally, the Secrets box on Streamlit Cloud) or an environment variable
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:          # no secrets file at all
        pass
    return os.environ.get("OPENROUTER_API_KEY", "")


def match_resume(api_key, model, resume_text, job_text):
    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(resume_text, job_text)}]
    for attempt in (1, 2):
        response = client.chat.completions.create(model=model, messages=messages, temperature=0)
        st.session_state["calls"] = st.session_state.get("calls", 0) + 1
        reply = response.choices[0].message.content or ""
        try:
            result = clean_result(extract_json(reply))
            break
        except (ValueError, json.JSONDecodeError):
            if attempt == 2:
                raise ValueError("The model did not return valid JSON twice in a row. Please try again.")
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "That was not valid JSON. Reply again with the JSON only."}]
    result["model_used"] = response.model
    return result


# ------------------------------------------------------------------ samples
@st.cache_data
def load_samples():
    if os.path.exists("samples.json"):
        with open("samples.json", encoding="utf-8") as f:
            return json.load(f)
    return None


samples = load_samples()


def load_sample(kind, key):
    field = "resume_text" if kind == "resumes" else "job_text"
    st.session_state[field] = samples[kind][key]["text"]
    st.session_state.pop("result", None)


def read_pdf():
    uploaded = st.session_state.get("resume_pdf")
    if uploaded is not None:
        text = "\n".join(page.extract_text() or "" for page in PdfReader(uploaded).pages).strip()
        st.session_state["resume_text"] = text
        st.session_state["pdf_note"] = (f"Read {len(text.split())} words from {uploaded.name}." if text
                                        else "No text found in that PDF (it may be a scanned image).")
        st.session_state.pop("result", None)


# ------------------------------------------------------------------ sidebar
configured_key = get_configured_key()
with st.sidebar:
    st.header("Settings")
    if configured_key:
        api_key = configured_key
        st.success("API key loaded from secrets")
    else:
        # Keep this box on the page on every run: Streamlit forgets a widget's value once it stops being shown
        api_key = st.text_input("Paste your OpenRouter API key", type="password", key="typed_key",
                                help="Kept only for this browser session. For a deployed app, "
                                     "add OPENROUTER_API_KEY in Secrets instead.").strip()
        if api_key:
            st.success("Key entered for this session")
        else:
            st.warning("No API key yet. Paste one above, or add OPENROUTER_API_KEY to secrets.")
    model = st.text_input("Model", value=DEFAULT_MODEL,
                          help="openrouter/free picks an available free model. "
                               "Or paste a specific model ID ending in :free from openrouter.ai/models.")
    redact = st.checkbox("Remove email and phone numbers before sending", value=True)
    calls_slot = st.empty()        # filled in at the end, after any new call
    st.divider()
    st.caption("Free models may log what you send. Use sample or anonymised resumes, "
               "and keep the contact-removal box ticked.")

# ------------------------------------------------------------------ main page
st.title("📄 Resume Matcher")
st.write("Compare a resume with a job description. A free LLM scores the match and suggests improvements.")

st.session_state.setdefault("resume_text", "")
st.session_state.setdefault("job_text", "")
left, right = st.columns(2)
with left:
    st.subheader("Resume")
    if samples:
        choice = st.selectbox("Load a sample resume", list(samples["resumes"]), index=None,
                              format_func=lambda k: samples["resumes"][k]["label"], key="sample_resume",
                              placeholder="Choose a sample...")
        if choice and st.session_state.get("last_resume") != choice:
            st.session_state["last_resume"] = choice
            load_sample("resumes", choice)
    st.file_uploader("...or upload a PDF", type="pdf", key="resume_pdf", on_change=read_pdf)
    if st.session_state.get("pdf_note"):
        st.caption(st.session_state["pdf_note"])
    st.text_area("Resume text", key="resume_text", height=320)
with right:
    st.subheader("Job description")
    if samples:
        choice = st.selectbox("Load a sample job", list(samples["jobs"]), index=None,
                              format_func=lambda k: samples["jobs"][k]["label"], key="sample_job",
                              placeholder="Choose a sample...")
        if choice and st.session_state.get("last_job") != choice:
            st.session_state["last_job"] = choice
            load_sample("jobs", choice)
    st.text_area("Job description text", key="job_text", height=390)

if st.button("Match", type="primary", key="match"):
    resume_text, job_text = st.session_state["resume_text"].strip(), st.session_state["job_text"].strip()
    if not api_key:
        st.error("Add your OpenRouter API key first (see the sidebar).")
    elif len(resume_text.split()) < 30 or len(job_text.split()) < 20:
        st.warning("Please give a full resume (30+ words) and a full job description (20+ words).")
    else:
        sent_resume = redact_contacts(resume_text) if redact else resume_text
        start = time.perf_counter()
        try:
            with st.spinner("The LLM is reading both documents (usually 5-30 seconds)..."):
                result = match_resume(api_key, model, sent_resume, job_text)
            result["seconds"] = time.perf_counter() - start
            result["unsupported"] = unsupported_skills(result, resume_text)
            st.session_state["result"] = result
        except openai.AuthenticationError:
            st.error("The API key was rejected. Check OPENROUTER_API_KEY in your secrets.")
        except openai.RateLimitError:
            st.error("Free limit reached (20 per minute or 50 per day). Wait a minute, or try again tomorrow.")
        except openai.NotFoundError:
            st.error(f"Model '{model}' was not found. Use openrouter/free or check the ID on openrouter.ai/models.")
        except (openai.APIError, ValueError) as error:
            st.error(f"Something went wrong: {error}")

result = st.session_state.get("result")
if result:
    st.divider()
    score = result["match_score"]
    c1, c2 = st.columns([1, 3])
    c1.metric("Match score", f"{score}/100")
    c2.progress(score / 100)
    message = f"**{result['verdict'] or 'Result'}.** {result['summary']}"
    if score >= 75:
        c2.success(message)
    elif score >= 40:
        c2.warning(message)
    else:
        c2.error(message)

    a, b = st.columns(2)
    a.markdown("#### ✅ Matched skills")
    a.markdown("\n".join(f"- {s}" for s in result["matched_skills"]) or "_None found_")
    b.markdown("#### ❌ Missing skills")
    b.markdown("\n".join(f"- {s}" for s in result["missing_skills"]) or "_None_")
    a.markdown("#### 💪 Strengths")
    a.markdown("\n".join(f"- {s}" for s in result["strengths"]) or "_None listed_")
    b.markdown("#### 💡 How to improve the resume")
    b.markdown("\n".join(f"- {s}" for s in result["suggestions"]) or "_None listed_")

    if result["unsupported"]:
        st.info("Check these 'matched' skills: they do not appear in the resume text, so the model may have "
                "inferred or invented them: " + ", ".join(result["unsupported"]))
    st.caption(f"Answered by {result['model_used']} in {result['seconds']:.0f} s. "
               "LLM scores are estimates and can change between runs; use them as feedback, not a final decision.")

calls_slot.caption(f"API calls this session: {st.session_state.get('calls', 0)} "
                   "(free plan: 20 per minute, 50 per day per account)")
