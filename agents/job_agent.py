import requests
from bs4 import BeautifulSoup
from data.sample_jobs import SAMPLE_JOBS


def scrape_remoteok():
    jobs = []

    try:
        url = "https://remoteok.com/remote-dev-jobs"
        headers = {"User-Agent": "Mozilla/5.0"}

        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        rows = soup.select("tr.job")

        for row in rows[:10]:
            title = row.select_one("h2")
            company = row.select_one(".companyLink h3")
            link = row.get("data-href")

            if title:
                jobs.append({
                    "title": title.text.strip(),
                    "company": company.text.strip() if company else "Unknown",
                    "location": "Remote",
                    "description": title.text.strip(),
                    "url": f"https://remoteok.com{link}" if link else "#",
                    "source": "remoteok"
                })

    except Exception:
        return []

    return jobs


def load_jobs():
    scraped = scrape_remoteok()

    if scraped:
        return scraped

    return SAMPLE_JOBS


def get_job_text(job):
    return f"""
    Title: {job.get('title', '')}
    Company: {job.get('company', '')}
    Location: {job.get('location', '')}
    Description: {job.get('description', '')}
    """.strip()
