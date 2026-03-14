from data.sample_jobs import SAMPLE_JOBS


def load_jobs():
    """
    Job Agent:
    Loads available jobs into the system.
    For now, this uses sample jobs.
    Later, this can be replaced with real web scraping.
    """
    return SAMPLE_JOBS


def get_job_text(job):
    """
    Combine job fields into one text block for matching.
    """
    return f"""
    Title: {job.get('title', '')}
    Company: {job.get('company', '')}
    Location: {job.get('location', '')}
    Description: {job.get('description', '')}
    """.strip()
