# Resume Matcher

A Streamlit app that compares a resume with a job description using a free LLM through [OpenRouter](https://openrouter.ai). It returns a match score, matched and missing skills, strengths and improvement tips. Built as a class project.

**Live app:** _paste your streamlit.app link here after deploying_

## How it works
1. Emails and phone numbers are removed from the resume.
2. The resume and job description go into one prompt with a scoring rubric and an exact JSON format.
3. The reply is parsed and checked; if it is not valid JSON, the app asks the model once more.
4. "Matched" skills that do not appear in the resume text are flagged as possibly invented.

## Files
| File | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Library versions |
| `samples.json` | 5 invented resumes and 3 invented job descriptions |
| `.streamlit/secrets.toml.example` | Shows the format of the secrets file (no real key) |

## Run it on your laptop
```bash
pip install -r requirements.txt
```
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste your key:
```toml
OPENROUTER_API_KEY = "sk-or-..."
```
Then run `streamlit run app.py`. `secrets.toml` is in `.gitignore`, so it never goes to GitHub.

## Deploy on Streamlit Community Cloud
Create the app from this repo, then in **Advanced settings → Secrets** paste the same line:
`OPENROUTER_API_KEY = "sk-or-..."`

## Limits and privacy
- OpenRouter free models: 20 requests per minute and 50 per day per account.
- Free model providers may log prompts. Use sample or anonymised resumes.
- LLM scores are estimates that can change between runs; use them as feedback, not as a hiring decision.
