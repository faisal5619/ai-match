# AI-Powered CV Screening and Job Matching System

**[Try it live →](https://ai-match-chsfunjq7ybea5wzqr3p4b.streamlit.app/)**

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
- **Scores the match** between the two using semantic similarity, so it recognises
  related wording rather than only exact keyword hits.
- **Identifies skill gaps** — the requirements the candidate does not yet meet — and
  returns that as concrete feedback rather than a pass/fail.
- **Recommends** the roles a given CV fits best.

## How it works

The system is split into four agents, each with one responsibility:

| Agent | Responsibility |
|---|---|
| `cv_agent.py` | Parses the CV and extracts structured skills and experience |
| `job_agent.py` | Parses the job description and extracts requirements |
| `match_agent.py` | Embeds both sides and scores the similarity between them |
| `recommendation_agent.py` | Ranks jobs for a candidate and reports missing skills |

Text from both the CV and the job description is converted into embeddings with
`sentence-transformers`, so two phrases that mean the same thing score as similar even
when the words differ. Embeddings are indexed with FAISS, which keeps matching fast as
the number of jobs grows.

## Tech stack

- **Python**
- **Streamlit** — web interface
- **sentence-transformers** — semantic embeddings
- **FAISS** — vector similarity search
- **scikit-learn** — supporting ML utilities
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
