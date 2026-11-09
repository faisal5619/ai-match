import html
import re
import time
import requests
from agents.cv_agent import extract_skills
from data.sample_jobs import SAMPLE_JOBS

REMOTEOK_API = "https://remoteok.com/api"
MAX_LIVE_JOBS = 20
# With fewer known skills a post can't be compared fairly: having 2 of 2 would count as a full match.
MIN_KNOWN_SKILLS = 4
MAX_DESCRIPTION_CHARS = 10000  # requirements often come last, so keep most of the post
LIVE_REFRESH_SECONDS = 3600

# The feed's tags are noisy (sales roles carry "engineer"), so roles are picked by title.
TECH_TITLE = re.compile(
    r"\b(developer|engineer|engineering|programmer|software|devops|sre|data scien\w+|data analyst|machine learning|ml|ai|"
    r"security|cloud|backend|back-end|frontend|front-end|full[- ]stack)\b",
    re.IGNORECASE
)

# Fetched at most once an hour, for every visitor: it's public job data, and it
# spares both Remote OK and this free server a request on every analysis.
_live = {"jobs": [], "fetched_at": None}


def clean_text(text):
    """The feed sends HTML, and characters like curly quotes garbled by a wrong encoding."""
    text = re.sub(
        "[\xc2-\xdf][\x80-\xbf]|[\xe0-\xef][\x80-\xbf]{2}",
        lambda m: m[0].encode("latin-1").decode("utf-8", "ignore"),
        text or ""
    )
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub("[�\x80-\x9f]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_remoteok():
    """
    Live remote jobs from Remote OK's public API. Its terms ask for a link back to
    each job and a mention of Remote OK as the source; the job cards do both.
    """
    jobs = []

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(REMOTEOK_API, headers=headers, timeout=10)

        # The first item is the API's legal notice, not a job.
        for item in r.json()[1:]:
            title = clean_text(item.get("position"))
            url = item.get("url") or ""

            # Only links to Remote OK itself are shown as job links.
            if not url.lower().startswith("https://remoteok.com/"):
                continue

            job = {
                "title": title,
                "company": clean_text(item.get("company")) or "Unknown",
                "location": clean_text(item.get("location")) or "Remote",
                "description": clean_text(item.get("description"))[:MAX_DESCRIPTION_CHARS],
                "url": url,
                "source": "remoteok"
            }

            if TECH_TITLE.search(title) and len(extract_skills(get_job_text(job))) >= MIN_KNOWN_SKILLS:
                jobs.append(job)

            if len(jobs) == MAX_LIVE_JOBS:
                break

    except Exception:
        return []

    return jobs


def live_jobs():
    fetched_at = _live["fetched_at"]
    if fetched_at is None or time.monotonic() - fetched_at > LIVE_REFRESH_SECONDS:
        _live["jobs"] = fetch_remoteok()
        _live["fetched_at"] = time.monotonic()
    return _live["jobs"]


def load_jobs():
    return live_jobs() + SAMPLE_JOBS


def get_job_text(job):
    return f"""
    Title: {job.get('title', '')}
    Company: {job.get('company', '')}
    Location: {job.get('location', '')}
    Description: {job.get('description', '')}
    """.strip()
