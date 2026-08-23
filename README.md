# AI-Powered CV Screening and Job Matching System

**[Try it live →](https://faisal-ai-match.streamlit.app/)**

No sign-up needed — click **Try the demo →** on the sign-in screen, or use
`demo@aimatch.app` / `demo1234`.

*Hosted on Streamlit Community Cloud's free tier, which hibernates apps after 12 hours
without traffic. If it's asleep, the page offers a wake button and takes about 30
seconds to start.*

An NLP recruitment tool that matches candidate CVs against job descriptions, scores
how well they fit, and tells the candidate which skills they are missing.

Built as my graduation project for the BSc Computer Science (Artificial Intelligence
concentration) at King Khalid University.

## What it does

- **Reads a CV** from PDF or Word and extracts the candidate's skills and experience.
- **Reads a job description** and extracts what the role actually requires.
- **Scores the match** between the two, combining how many required skills the candidate
  actually has with how similar the two documents read overall.
- **Identifies skill gaps** — the requirements the candidate does not yet meet — and
  returns that as concrete feedback rather than a pass/fail.
- **Recommends** the roles a given CV fits best.

## How it works

The system is split into four agents, each with one responsibility:

| Agent | Responsibility |
|---|---|
| `cv_agent.py` | Parses the CV and extracts structured skills and experience |
| `job_agent.py` | Parses the job description and extracts requirements |
| `match_agent.py` | Scores a shortlisted job against the CV and produces the final number |
| `recommendation_agent.py` | Ranks jobs for a candidate and reports missing skills |

Matching runs in two stages, and the distinction matters.

**Stage 1 — retrieval.** Every job description is converted into an embedding with
`sentence-transformers` (`all-MiniLM-L6-v2`) and indexed in FAISS (`IndexFlatL2`). The CV
is embedded the same way, and FAISS returns the five nearest jobs. This stage recognises
related wording rather than exact keywords, and it keeps the work manageable as the number
of jobs grows — only five candidates get scored in detail rather than all of them.

**Stage 2 — ranking.** Each shortlisted job is then scored two ways:

| Component | Weight | What it measures |
|---|---:|---|
| Skill overlap | 75% | how many of the job's required skills appear in the CV |
| TF-IDF cosine similarity | 25% | how similar the two documents are as text |

So the embeddings decide *which* jobs get considered, and the final score is driven mostly
by concrete skill overlap. Semantic similarity alone produced scores that were too close
together to be useful — unrelated jobs came out looking comparable — which is why the
scoring falls back to something more literal.

## Tech stack

- **Python**
- **Streamlit** — web interface
- **sentence-transformers** — embeddings for the retrieval stage
- **FAISS** — vector index for shortlisting candidate jobs
- **scikit-learn** — TF-IDF and cosine similarity for the scoring stage
- **PyPDF2 / python-docx** — reading CV files
- **BeautifulSoup** — parsing scraped job pages

## Running it locally

```bash
git clone https://github.com/faisal5619/ai-match.git
cd ai-match
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

> First run downloads the sentence-transformers model, so give it a moment.

## Project structure

```
ai-match/
├── app.py                        # Streamlit interface and app flow
├── auth_db.py                    # User authentication
├── agents/
│   ├── cv_agent.py               # CV parsing and skill extraction
│   ├── job_agent.py              # Job description parsing
│   ├── match_agent.py            # Semantic matching and scoring
│   └── recommendation_agent.py   # Job ranking and skill-gap analysis
├── database/
│   ├── db.py                     # Database layer
│   └── vector_agent.py           # Vector index handling
├── data/
│   └── sample_jobs.py            # Sample job postings for the demo
└── requirements.txt
```

## Status

Local demo, not deployed. Runs on sample job data.

## Author

**Faisal Alhudaithy** — BSc Computer Science (AI concentration), King Khalid University
[GitHub](https://github.com/faisal5619) · [LinkedIn](https://www.linkedin.com/in/faisal-alhudaithy/)
