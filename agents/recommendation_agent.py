def generate_recommendations(match_results, top_n=3):
    """
    Recommendation Agent:
    Takes ranked match results and returns the top recommended jobs
    with short explanations.
    """

    recommendations = []

    top_jobs = match_results[:top_n]

    for job in top_jobs:
        matched = job.get("matched_skills", [])
        missing = job.get("missing_skills", [])

        if job["final_score"] >= 75:
            fit_label = "Strong Match"
        elif job["final_score"] >= 50:
            fit_label = "Moderate Match"
        else:
            fit_label = "Weak Match"

        if matched:
            match_reason = f"Your CV aligns with key skills such as: {', '.join(matched[:5])}."
        else:
            match_reason = "Your CV has limited overlap with this job's detected skills."

        if missing:
            improvement_tip = f"Consider improving or highlighting these skills: {', '.join(missing[:5])}."
        else:
            improvement_tip = "You already cover the major detected skills for this role."

        recommendations.append({
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "final_score": job["final_score"],
            "fit_label": fit_label,
            "reason": match_reason,
            "tip": improvement_tip,
            "url": job["url"],
            "matched_skills": matched,
            "missing_skills": missing
        })

    return recommendations
