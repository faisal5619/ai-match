import re
import math
import json
import time
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
import streamlit as st
from agents.cv_agent import analyze_cv, fetch_github_skills, SKILLS as MATCH_SKILLS
from agents.job_agent import load_jobs
from agents.match_agent import analyze_all_jobs
from agents.recommendation_agent import generate_recommendations

# ---------------- CONFIG ----------------
ASSETS = Path(__file__).parent / "assets"
# The tab icon uses the small-size cut of the logo, drawn for 16-32 px.
st.set_page_config(page_title="AI Match", page_icon=str(ASSETS / "favicon.png"), layout="wide")

# The top-bar logo is the vector mark in assets/. Its fill comes from the
# theme (--logo-fill): the sunset gradient in light mode, plain light grey in dark.
LOGO_SVG = re.sub(r"<title>.*?</title>", "", (ASSETS / "logo.svg").read_text(encoding="utf-8")
                  ).replace('role="img" aria-label="AI Match"', 'aria-hidden="true"'
                  ).replace('fill="url(#aim-sunset)"', 'style="fill:var(--logo-fill)"')

# Saudi Arabia is UTC+3 all year (no daylight saving), so a fixed offset is
# enough and avoids needing timezone data installed on the server.
RIYADH = timezone(timedelta(hours=3))

# ---------------- SESSION DEFAULTS ----------------
# ---------------- VISIT STORAGE ----------------
# There are no accounts. What a visitor adds (their CV, preferences and best
# match) is held in server memory for their session only, and is cleared soon after
# they close the tab.
for _key, _default in {"cv_name": None, "cv_bytes": None, "best_match": None,
                       "profile": {"full_name": "", "preferred_location": "", "job_type": "",
                                   "field_of_interest": "", "github_username": ""}}.items():
    st.session_state.setdefault(_key, _default)


def load_user_cv():
    return st.session_state.cv_name, st.session_state.cv_bytes


def save_user_cv(name, data):
    st.session_state.cv_name, st.session_state.cv_bytes = name, data
    st.session_state.analysis = None  # its matches were for the previous CV


def load_user_profile():
    return st.session_state.profile


def save_user_profile(full_name, location, job_type, field, github):
    st.session_state.profile = {"full_name": full_name.strip(), "preferred_location": location, "job_type": job_type,
                                "field_of_interest": field, "github_username": github.strip()}
    # Matches already on screen were ranked with the old preferences: rank them again.
    if load_analysis():
        st.session_state.rerank = True


def load_analysis():
    """The latest analysis, or None. One made with another CV is out of date, so it's none."""
    analysis = st.session_state.get("analysis")
    return analysis if analysis and analysis["cv_name"] == st.session_state.cv_name else None


def load_best_match():
    """The top role from the latest match, or None. A match made with another CV is out of date, so it's none."""
    best = st.session_state.best_match
    return best if best and best["cv_name"] == st.session_state.cv_name else None


def save_best_match(title, company, score):
    st.session_state.best_match = {"title": title, "company": company, "score": score,
                                   "cv_name": st.session_state.cv_name}

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- HELPERS ----------------
def cv_skills(cv_name: str, cv_bytes: bytes) -> list:
    """
    Skills the matcher detects in a CV. Home shows them on every load, so they're
    remembered with the visit. Not st.cache_data: that is shared and outlives the visit.
    """
    memo = st.session_state.get("cv_skills_memo")
    if memo and memo["cv"] == (cv_name, cv_bytes):
        return memo["skills"]
    try:
        skills = analyze_cv(cv_name, cv_bytes)["skills"]
    except Exception:
        skills = []
    st.session_state.cv_skills_memo = {"cv": (cv_name, cv_bytes), "skills": skills}
    return skills

# Words nearly every CV has, in English and Arabic. Checked here rather than in
# agents/cv_agent.py, because a push to agents/ needs a manual reboot to take effect.
CV_WORDS_EN = re.compile(r"\b(education|experience|employment|university|college|bachelor|bachelors|master|masters|"
                         r"degree|diploma|gpa|internship|certifications?|curriculum vitae|resume|work history|"
                         r"references|professional summary)\b")
CV_WORDS_AR = ("السيرة الذاتية", "جامعة", "بكالوريوس", "خبرة", "الخبرات", "مهارات", "المهارات", "التعليم",
               "المؤهلات", "الشهادات")

def looks_like_cv(text: str, skills: list) -> bool:
    """A deliberately gentle check: two CV words or three known skills is enough to pass."""
    lowered = text.lower()
    cv_words = set(CV_WORDS_EN.findall(lowered)) | {w for w in CV_WORDS_AR if w in text}
    return len(cv_words) >= 2 or len(skills) >= 3

def read_cv(cv_name: str, cv_bytes: bytes):
    """The CV's text and skills, or (None, a plain message saying why it can't be matched)."""
    try:
        cv_data = analyze_cv(cv_name, cv_bytes)
    except Exception:
        return None, ("We couldn't open this file. It may be damaged or password-protected. "
                      "Save it again as a PDF, Word (.docx) or text file and upload it once more.")
    if not cv_data["text"].strip():
        return None, ("We couldn't find any text in this file. If it's a scanned image or a photo, "
                      "upload the original PDF or Word file instead.")
    if not looks_like_cv(cv_data["text"], cv_data["skills"]):
        return None, ("This file doesn't look like a CV, so we didn't rank it. We look for things like education, "
                      "work experience and skills. Upload your CV as a PDF, Word (.docx) or text file.")
    return cv_data, None

def is_example(job) -> bool:
    """The curated roles are written for the demo; only Remote OK listings are real openings."""
    return job.get("source") != "remoteok"

def source_tag(job) -> str:
    return " · Example role" if is_example(job) else " · Live job from Remote OK"

def is_senior(job) -> bool:
    """The score only compares skills, so a senior title is flagged rather than scored."""
    return bool(re.search(r"\b(senior|sr|staff|principal|lead|head|director)\b", job["title"], re.IGNORECASE))

def seniority_tag(job) -> str:
    return " · Senior role" if is_senior(job) else ""

def preference_tag(result) -> str:
    """Preferences re-order the roles but aren't part of the fit score, so their points are shown beside it."""
    points = round(result.get("boosted_score", result["final_score"]) - result["final_score"])
    return f" · +{points} from your preferences" if points > 0 else ""

def job_link_label(job) -> str:
    # An example role has no listing of its own; its link searches for similar real jobs.
    return "Find similar jobs" if is_example(job) else "View on Remote OK"

def chips(skills, kind="have"):
    # --i is each chip's position, so the results reveal can stagger them.
    return "".join(f'<span class="chip chip-{kind}" style="--i:{i}">{escape(s)}</span>'
                   for i, s in enumerate(skills))

def first_name_of(name: str) -> str:
    return name.split()[0] if name.strip() else ""

# ---------------- LOOK ----------------
# Two colour sets the visitor can switch between from the top bar. style.css
# never names a colour itself; it reads these tokens, so a theme is just one of
# these dictionaries.
#
# The "carbon fibre" panel look comes from four tokens together: a surface that
# is lighter at the top (surface-top) and fades to surface, a fine grain laid
# over it, a hairline edge, and a brighter highlight on the top edge (edge-hi).
# Grain is a tiny SVG noise texture; it must stay free of ";" because the tokens
# are joined with ";".
_GRAIN = ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E"
          "%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' "
          "stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 {v} 0 0 0 0 {v} 0 0 0 0 {v} 0 0 0 {a} 0'/%3E"
          "%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")")
THEMES = {
    "dark": {
        "ground": "#08090a", "surface": "#101113", "surface-top": "#1b1c20", "raised": "#15161a",
        "field": "#0c0d0f",
        "border": "#24262b", "border-strong": "#3a3d45", "rule": "#1a1b1f", "control": "#34373e",
        "edge": "rgba(255,255,255,.08)", "edge-hi": "rgba(255,255,255,.07)",
        "grain": _GRAIN.format(v=1, a=".055"), "spot": "rgba(170,180,200,.10)",
        "hover": "rgba(255,255,255,.06)", "active": "rgba(255,255,255,.10)",
        "text": "#ebecee", "muted": "#9a9ea6",
        "accent": "#7ea4ff", "accent-soft": "#41598f", "accent-tint": "rgba(126,164,255,.15)",
        "accent-ink": "#c5d5fb", "on-accent": "#08090a",
        "primary-bg": "#ebecee", "primary-fg": "#08090a", "negative": "#f08a80",
        "gold": "#e2b44f", "gold-tint": "rgba(226,180,79,.14)", "gold-ink": "#f4d996",
        "gold-surface": "#13110b", "gold-surface-top": "#221d10", "gold-border": "rgba(226,180,79,.45)",
        "glow": "rgba(126,164,255,.16)", "shadow": "rgba(0,0,0,.7)", "shadow-soft": "rgba(0,0,0,.45)",
        "track": "#212227", "chip-muted-bg": "rgba(255,255,255,.05)", "chip-muted-ink": "#b9bcc3",
        # Small coloured icons, one hue per subject, used sparingly.
        "c-blue": "#6ea8fe", "c-gold": "#e8b84a", "c-green": "#4fd08f", "c-orange": "#f39a5a",
        "c-pink": "#ef84b0", "c-violet": "#a395ff",
        # Light-only extras (see the light theme); off here.
        "wash-a": "0%", "wash-base": "transparent", "wash-pad": "0", "wash-min": "0", "corner": "flex", "lift": "transparent",
        "ground-wash": "none", "icon-tint": "0%", "icon-pad": "0px", "wash-hover": "1", "bar-bleed": "0px",
        "scroll-thumb": "rgba(231,232,234,.4)", "logo-fill": "#ebecee", "veil": "rgba(16,17,19,.82)", "veil-pad": "24px 28px",
        # The burst behind every page, in greys: ray colours centre to tip, then haze.
        "burst": "block", "burst-1": "#9a9fa9", "burst-2": "#6c707a", "burst-3": "#4a4d55",
        "burst-4": "#383b42", "burst-5": "rgba(150,160,180,.10)", "burst-6": "rgba(120,130,150,.05)",
        "burst-7": "#b9c2d6", "burst-8": "2",  # stars: soft blue-grey, not pure white; size
        "pill": "0%", "pill-edge": "var(--control)", "pill-edge-hover": "var(--border-strong)",
        "pill-on": "0%", "link-pad": "0", "panel-glow": "0%", "table-line": "rgba(42,44,49,.45)",
        "fit-ink": "65%",  # other matches' scores, muted towards white in their fit colour
    },
    # Stripe-style: white cards on a pale blue-grey page, thin soft borders, a
    # violet accent, and a soft colour wash behind each card's graphic.
    "light": {
        "ground": "#f6f8fb", "surface": "#ffffff", "surface-top": "#ffffff", "raised": "#ffffff",
        "field": "#ffffff",
        "border": "#e3e8ef", "border-strong": "#c7bff8", "rule": "#edf0f5", "control": "#d5dbe5",
        "edge": "#e3e8ef", "edge-hi": "transparent",
        "grain": "none", "spot": "rgba(122,100,255,.07)",
        "hover": "rgba(40,30,90,.05)", "active": "rgba(91,78,230,.10)",
        "text": "#0d1b33", "muted": "#55627a",
        "accent": "#5b4ee6", "accent-soft": "#b7b0f5", "accent-tint": "#eeebff",
        "accent-ink": "#3b2fb8", "on-accent": "#ffffff",
        "primary-bg": "#5b4ee6", "primary-fg": "#ffffff", "negative": "#c0362c",
        "gold": "#b07d12", "gold-tint": "#f7ebcb", "gold-ink": "#5a3f06",
        "gold-surface": "#fffaf0", "gold-surface-top": "#fffdf8", "gold-border": "#e6c878",
        "glow": "rgba(124,92,255,.16)", "shadow": "rgba(70,50,170,.30)", "shadow-soft": "rgba(50,50,93,.07)",
        "track": "#eceff4", "chip-muted-bg": "#f0f2f6", "chip-muted-ink": "#3c4659",
        "c-blue": "#2f6fd6", "c-gold": "#b7871c", "c-green": "#1d9a5f", "c-orange": "#e0662a",
        "c-pink": "#d4407e", "c-violet": "#5b4ee6",
        # The wash behind each card's graphic: how strong its two colours are, over
        # this base. Each card picks its own colours in style.css.
        "wash-a": "42%", "wash-base": "#fdfcff",
        "wash-pad": "22px", "wash-min": "200px", "corner": "flex", "lift": "rgba(50,50,93,.12)",
        # Soft colour blooms on the page itself, behind the cards.
        "ground-wash": ("radial-gradient(700px 420px at 8% 4%, rgba(140,110,255,.16), transparent 70%), "
                        "radial-gradient(640px 420px at 96% 10%, rgba(255,140,170,.14), transparent 70%), "
                        "radial-gradient(800px 500px at 60% 85%, rgba(255,170,90,.10), transparent 70%)"),
        # Each card's icon sits on a small tile tinted in its own colour.
        "icon-tint": "14%", "icon-pad": "8px",
        # How much brighter the wash gets on hover.
        "wash-hover": "1.3",
        # The top bar's frosted strip reaches the window edges, not just the content column.
        "bar-bleed": "100vmax",
        "scroll-thumb": "rgba(60,55,110,.35)", "logo-fill": "url(#aim-sunset)",
        # The sunset burst behind every page (drawn by FX_SCRIPT).
        "burst": "block", "burst-1": "#ffb27a", "burst-2": "#ff7aa6", "burst-3": "#d24ff2",
        "burst-4": "#5a3cf0", "burst-5": "rgba(255,170,190,.45)", "burst-6": "rgba(205,150,255,.28)",
        "burst-7": "rgba(110,80,230,.8)", "burst-8": "2",  # stars: soft violet; size
        # A frosted sheet under text that sits straight on the page, over the burst.
        "veil": "rgba(255,255,255,.82)", "veil-pad": "24px 28px",
        # Home card buttons: a pill tinted this much with the card's colour, no edge.
        "pill": "14%", "pill-edge": "transparent", "pill-edge-hover": "transparent",
        # Contact links become pills too; each panel gets a soft glow of its colour.
        "pill-on": "100%", "link-pad": "6px 14px", "panel-glow": "12%", "table-line": "#e4e8ef", "fit-ink": "70%",
    },
}
if "theme" not in st.session_state:
    st.session_state.theme = st.query_params.get("theme") if st.query_params.get("theme") in THEMES else "dark"

# A tiny script that runs in the page itself (Streamlit lets an embedded
# component reach its parent page). It does two things CSS alone can't:
#   - adds .in-view to each bento card when it scrolls into sight, so its
#     animation starts then and not on page load;
#   - feeds the pointer position to each card so its glow follows the mouse;
#   - draws the burst behind every page: rays fanning up from the bottom of the
#     window that sway, and stretch toward the pointer. Its colours come from the
#     theme (sunset in light, greys in dark); while hidden, nothing is drawn.
# It marks the page fx-ready first; without it, style.css hides nothing.
# (The code below sits inside a JS template literal: no backticks or dollar-braces.)
FX_SCRIPT = """
<script>
(function () {
  const w = window.parent, d = w.document;
  if (w.__aiMatchFx) return;
  w.__aiMatchFx = true;
  const code = `
    document.documentElement.classList.add('fx-ready');
    const io = new IntersectionObserver(entries => entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('in-view'); io.unobserve(e.target); }
    }), { threshold: 0.2 });
    const watch = () => document.querySelectorAll('[class*="st-key-bento"]:not([data-fx])').forEach(el => {
      el.dataset.fx = '1';
      io.observe(el);
      el.addEventListener('pointermove', ev => {
        const r = el.getBoundingClientRect();
        el.style.setProperty('--mx', (ev.clientX - r.left) + 'px');
        el.style.setProperty('--my', (ev.clientY - r.top) + 'px');
      });
    });
    watch();
    new MutationObserver(watch).observe(document.body, { childList: true, subtree: true });

    // Sunset burst. Each ray has a fixed angle and length, sways on its own
    // phase, and stretches toward the pointer when it points near it.
    const cv = document.createElement('canvas');
    cv.id = 'burst';
    cv.setAttribute('aria-hidden', 'true');
    const cx = cv.getContext('2d');
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const stars = Array.from({ length: 170 }, () => ({
      x: Math.random(), y: Math.random(), r: 0.4 + Math.random() * 0.9,
      o: 0.3 + 0.7 * Math.random(), ph: Math.random() * 6.28, sp: 0.5 + Math.random() * 1.5,
    }));
    const rays = Array.from({ length: 340 }, () => ({
      a: Math.PI * (0.96 + 1.08 * Math.random()),          // upward half-circle, a little past level
      f: 0.3 + 0.7 * Math.pow(Math.random(), 0.6),         // share of the radius
      op: 0.25 + 0.65 * Math.random(), lw: 0.6 + 0.9 * Math.random(),
      ph: Math.random() * 6.28, sp: 0.4 + 0.8 * Math.random(), ext: 0,
    }));
    let W = 0, H = 0, ptr = null, running = false, pal = [];
    const size = () => {
      const k = Math.min(devicePixelRatio || 1, 1.5);
      W = innerWidth; H = innerHeight;
      cv.width = W * k; cv.height = H * k;
      cx.setTransform(k, 0, 0, k, 0, 0);
      if (running && still) draw(0);                       // a still picture still fits the window
    };
    addEventListener('resize', size);
    addEventListener('pointermove', e => { ptr = [e.clientX, e.clientY]; });
    document.documentElement.addEventListener('mouseleave', () => { ptr = null; });
    const draw = t => {
      // The centre sits just above the bottom edge, so it shows below the last cards.
      const ox = W / 2, oy = H - 40, R = Math.min(H * 1.1, W * 0.65);
      let pa = 0, near = false;
      if (ptr && !still) {
        pa = Math.atan2(ptr[1] - oy, ptr[0] - ox);
        if (pa < 0) pa += 2 * Math.PI;
        near = Math.hypot(ptr[0] - ox, ptr[1] - oy) < R * 1.3;
      }
      cx.globalCompositeOperation = 'source-over';
      cx.clearRect(0, 0, W, H);
      for (const r of rays) {
        const pull = near ? Math.exp(-Math.pow((r.a - pa) / 0.13, 2)) : 0;
        r.ext += (pull * 0.22 - r.ext) * 0.08;               // ease toward the pointer's pull
        const len = R * (r.f * (1 + 0.035 * Math.sin(t * 0.001 * r.sp + r.ph)) + r.ext);
        const x = ox + Math.cos(r.a) * len, y = oy + Math.sin(r.a) * len;
        cx.globalAlpha = Math.min(1, r.op + pull * 0.4);
        cx.lineWidth = r.lw;
        cx.beginPath(); cx.moveTo(ox, oy); cx.lineTo(x, y); cx.stroke();
        cx.beginPath(); cx.arc(x, y, r.lw + 0.6, 0, 6.28); cx.fill();
      }
      cx.globalAlpha = 1;
      // Colour the rays by distance from the origin: peach, pink, magenta, violet.
      cx.globalCompositeOperation = 'source-in';
      const g = cx.createRadialGradient(ox, oy, 0, ox, oy, R * 1.15);
      g.addColorStop(0, pal[0]); g.addColorStop(0.35, pal[1]);
      g.addColorStop(0.65, pal[2]); g.addColorStop(1, pal[3]);
      cx.fillStyle = g; cx.fillRect(0, 0, W, H);
      // A soft haze behind them.
      cx.globalCompositeOperation = 'destination-over';
      const h = cx.createRadialGradient(ox, oy, 0, ox, oy, R * 1.2);
      h.addColorStop(0, pal[4]); h.addColorStop(0.5, pal[5]);
      h.addColorStop(1, 'rgba(0,0,0,0)');
      cx.fillStyle = h; cx.fillRect(0, 0, W, H);
      // Fade it out toward the top, so page headings stay easy to read.
      cx.globalCompositeOperation = 'destination-in';
      const fade = cx.createLinearGradient(0, 0, 0, H);
      fade.addColorStop(0, 'rgba(0,0,0,.2)'); fade.addColorStop(0.32, 'rgba(0,0,0,1)');
      cx.fillStyle = fade; cx.fillRect(0, 0, W, H);
      // Small twinkling stars, in the theme's star colour ("none" turns them off).
      cx.globalCompositeOperation = 'source-over';
      if (pal[6] !== 'none') {
        cx.fillStyle = pal[6];
        for (const s of stars) {
          cx.globalAlpha = s.o * (0.55 + 0.45 * Math.sin(t * 0.001 * s.sp + s.ph));
          cx.beginPath(); cx.arc(s.x * W, s.y * H, s.r * pal[7], 0, 6.28); cx.fill();
        }
        cx.globalAlpha = 1;
      }
    };
    const loop = t => {
      if (!running) return;
      draw(t);
      if (!still) requestAnimationFrame(loop);
    };
    // Twice a second: keep the canvas in place (Streamlit can rebuild the app
    // frame) and start or stop drawing as the theme shows or hides it.
    setInterval(() => {
      const app = document.querySelector('.stApp');
      if (app && app.firstChild !== cv) { app.prepend(cv); size(); }
      const on = getComputedStyle(cv).display !== 'none';
      // Ray colours (centre to tip) and haze colours come from the theme.
      const css = getComputedStyle(document.documentElement);
      pal = [1, 2, 3, 4, 5, 6, 7, 8].map(i => css.getPropertyValue('--burst-' + i).trim());
      if (still && on) draw(0);
      if (on && !running) { running = true; requestAnimationFrame(loop); }
      if (!on) running = false;
    }, 500);
  `;
  const s = d.createElement('script');
  s.textContent = code;
  d.head.appendChild(s);
})();
</script>
"""


def apply_look():
    """Put the current theme's tokens and style.css on the page, plus the effects script."""
    tokens = ";".join(f"--{name}:{value}" for name, value in THEMES[st.session_state.theme].items())
    css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>:root{{{tokens}}}\n{css}</style>", unsafe_allow_html=True)
    with st.container(key="fx_host"):
        st.iframe(FX_SCRIPT, height=1)


def theme_toggle(key):
    """A button that flips between the dark and light colour sets."""
    dark = st.session_state.theme == "dark"
    if st.button("Light" if dark else "Dark", key=key,
                 icon=":material/light_mode:" if dark else ":material/dark_mode:"):
        st.session_state.theme = "light" if dark else "dark"
        st.query_params["theme"] = st.session_state.theme
        st.rerun()


apply_look()


# ════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE
# ════════════════════════════════════════════════════════════════════════════
LINKEDIN_URL = "https://www.linkedin.com/in/faisal-alhudaithy/"

# Written for Saudi Arabia's Personal Data Protection Law (PDPL): what is kept,
# why, who else sees it, where it is processed, for how long, and the user's rights.
# Shown on the Privacy page.
PRIVACY_HTML = f"""
<div class="prose">
<p>AI Match is a student project by Faisal Alhudaithy. There are no accounts: you use it without
signing up, and this page explains what happens to what you add. Last updated 9 November 2026.</p>

<h2>What it keeps</h2>
<ul>
<li>The CV you upload, the preferences and name you enter, a GitHub username if you add one, and your
latest best match. These are sent to the server that runs this site and held in its memory for your
visit only. They are cleared soon after you close the tab.</li>
<li>AI Match saves none of it after that: there is no database of users or CVs.</li>
</ul>

<h2>Why</h2>
<p>Only to match your CV against job descriptions and show you the results. Nothing is sold, there are
no ads and no emails. AI Match itself adds no analytics or tracking, but its host does (see below).</p>

<h2>Who else handles it</h2>
<ul>
<li><strong>Streamlit Community Cloud</strong> hosts the app, so your CV is processed on its servers,
outside Saudi Arabia, while you use it. Streamlit also sets its own cookies and runs its own analytics
(Google Analytics, Segment, Heap and Datadog) on every app it hosts. AI Match can't turn these off; see
<a class="link" href="https://streamlit.io/privacy-policy" target="_blank" rel="noopener">Streamlit's privacy policy</a>.</li>
<li><strong>Google Fonts</strong> supplies the site's fonts, so your browser fetches them from Google,
which sees your IP address
(<a class="link" href="https://policies.google.com/privacy" target="_blank" rel="noopener">Google's privacy policy</a>).</li>
<li><strong>GitHub:</strong> if you add a username, the app reads the languages of your public
repositories from GitHub.</li>
<li><strong>Remote OK</strong> supplies the live job listings. Nothing about you is sent to it.</li>
</ul>

<h2>How long it is kept</h2>
<p>Only for your visit. Everything you add is cleared soon after you close the tab.</p>

<h2>Your rights</h2>
<ul>
<li>Change or remove your CV and preferences at any time on the Profile page, or close the tab.</li>
<li>For any question about your data, message Faisal on
<a class="link" href="{LINKEDIN_URL}" target="_blank" rel="noopener">LinkedIn</a>.</li>
</ul>

<h2>Age</h2>
<p>AI Match is meant for people aged 18 and over.</p>
</div>
"""


# ════════════════════════════════════════════════════════════════════════════
#  NAV / ROUTING
# ════════════════════════════════════════════════════════════════════════════
PAGES = ["Home", "Dashboard", "Profile", "About", "Contact", "Privacy"]

qp = st.query_params.get("page")
if qp in PAGES:
    st.session_state.page = qp

first_name = first_name_of(load_user_profile()["full_name"])
current_page = st.session_state.page

def set_page(p: str):
    st.session_state.page = p
    st.query_params["page"] = p
    st.rerun()

# ---------------- TOP BAR ----------------
# Brand on the left, pages in the middle, the theme switch on the right. "About"
# opens a small menu on hover: the menu is a container style.css hides until its
# group is hovered or focused. Touch screens get a row instead (see subnav_about).
def nav_btn(label, target, key=None, icon=None):
    if st.button(label, key=key or f"nav_{target}", icon=icon,
                 type="primary" if current_page == target else "secondary"):
        set_page(target)


with st.container(key="topbar", horizontal=True, vertical_alignment="center", gap="small"):
    st.markdown(
        f'<div class="brand">{LOGO_SVG}<span>AI Match</span></div>',
        unsafe_allow_html=True
    )
    with st.container(key="navlinks", horizontal=True, vertical_alignment="center", gap=None):
        nav_btn("Home", "Home", icon=":material/home:")
        nav_btn("Matches", "Dashboard", icon=":material/emoji_events:")
        nav_btn("Profile", "Profile", icon=":material/person:")
        with st.container(key="navgroup_about", width="content", gap=None):
            if st.button("About", key="nav_About", icon=":material/expand_more:", icon_position="right",
                         type="primary" if current_page in ("About", "Contact", "Privacy") else "secondary"):
                set_page("About")
            with st.container(key="navmenu_about", gap=None):
                nav_btn("How it works", "About", key="menu_About", icon=":material/info:")
                nav_btn("Contact", "Contact", key="menu_Contact", icon=":material/mail:")
                nav_btn("Privacy", "Privacy", key="menu_Privacy", icon=":material/shield:")
    with st.container(key="navright", horizontal=True, vertical_alignment="center", gap="small", width="content"):
        theme_toggle("theme_top")

# A touch screen can't hover, so the About menu never opens there. Instead its
# pages get their own row under the bar while one of them is open; style.css
# shows this row only on phones and touch screens.
if current_page in ("About", "Contact", "Privacy"):
    with st.container(key="subnav_about", horizontal=True, vertical_alignment="center", gap=None):
        nav_btn("How it works", "About", key="sub_About")
        nav_btn("Contact", "Contact", key="sub_Contact")
        nav_btn("Privacy", "Privacy", key="sub_Privacy")

# ════════════════════════════════════════════════════════════════════════════
#  PAGES
# ════════════════════════════════════════════════════════════════════════════

LOCATION_OPTIONS = ["", "Riyadh", "Jeddah", "Abha", "Dammam", "Medina", "Remote", "Any"]
JOB_TYPE_OPTIONS = ["", "Full-time", "Part-time", "Internship", "Remote", "Freelance"]
FIELD_OPTIONS = ["", "Artificial Intelligence", "Web Development", "Data Science", "Cybersecurity",
                 "Mobile Development", "Cloud Computing", "Software Engineering", "DevOps", "Other"]


# Line icons (24px grid, drawn with the current text colour) for card and panel
# titles. Each subject keeps one colour everywhere it appears.
ICONS = {
    "sparkles": '<path d="M12 3l1.8 4.9L19 9.7l-5.2 1.8L12 16.4l-1.8-4.9L5 9.7l5.2-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
    "trophy": '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4z"/><path d="M17 5h3v2a3 3 0 0 1-3 3M7 5H4v2a3 3 0 0 0 3 3"/>',
    "bars": '<path d="M5 20v-6M12 20V8M19 20V4M3 20h18"/>',
    "file": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/>',
    "sliders": '<path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1"/><circle cx="15" cy="6" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="18" r="2"/>',
    "cv": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>'
          '<circle cx="9.5" cy="11" r="2"/><path d="M6.5 17c.5-1.5 1.6-2.3 3-2.3s2.5.8 3 2.3M15 14h2M15 17h2"/>',
    "zap": '<path d="M13 2L4 14h8l-1 8 9-12h-8z"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "bug": '<rect x="8" y="6" width="8" height="14" rx="4"/><path d="M12 10v10M8 12H4M20 12h-4M8 17H5M19 17h-3M9 6L7 3M15 6l2-3"/>',
    "code": '<path d="M8 8l-4 4 4 4M16 8l4 4-4 4M13.5 5l-3 14"/>',
    "chat": '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/>',
    "trash": '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    "compass": '<circle cx="12" cy="12" r="9"/><path d="M15.5 8.5l-2 5-5 2 2-5z"/>',
}


def icon(name, color):
    """An inline line icon in one of the theme's icon colours (blue, gold, green, ...)."""
    return (f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true" style="color:var(--c-{color})">{ICONS[name]}</svg>')


def panel_title(text, icon_name, color):
    return (f'<h2 class="panel-title"><span class="ico" style="color:var(--c-{color})">{icon(icon_name, color)}</span>'
            f'{text}</h2>')


def set_label_head(text, icon_name, color):
    """Opens a Profile section label (the caller closes the div) with its icon; the colour also tints its sheet."""
    return (f'<div class="set-label {color}"><h2><span class="ico" style="color:var(--c-{color})">'
            f'{icon(icon_name, color)}</span>{text}</h2>')


def bento_head(title, sub="", icon_name=None, color=None):
    icon_html = (f'<div class="bento-icon" style="color:var(--c-{color})">{icon(icon_name, color)}</div>'
                 if icon_name else "")
    sub_html = f'<p class="bento-sub">{sub}</p>' if sub else ""
    return f'{icon_html}<h2 class="bento-title">{title}</h2>{sub_html}'


def bento_open(key, target, label):
    """The small square button in a card's top-right corner (light theme only) that opens its page."""
    # The label is hidden on screen by style.css but still read out by screen readers.
    if st.button(f"Open {label}", key=f"open_{key}", icon=":material/north_east:", help=label):
        set_page(target)


def page_home():
    profile = load_user_profile()
    saved_name, saved_bytes = load_user_cv()
    found = sorted(cv_skills(saved_name, saved_bytes)) if saved_name else []
    not_found = sorted(set(MATCH_SKILLS) - set(found))
    last = load_best_match()

    now = datetime.now(RIYADH)
    part = "morning" if now.hour < 12 else "afternoon" if now.hour < 18 else "evening"
    if last:
        fact = (f"Your best match so far is <strong>{escape(last['title'])}</strong> at "
                f"{escape(last['company'])}, at <strong>{last['score']:.0f}%</strong>.")
    elif saved_name:
        fact = f"<strong>{escape(saved_name)}</strong> is saved for this visit."
    else:
        fact = "Upload a CV to see which roles it fits."

    with st.container(key="hero_home"):
        head, action = st.columns([9, 3], vertical_alignment="bottom")
        with head:
            st.markdown(
                f'<h1 class="greeting">Good {part}{", " + escape(first_name) if first_name else ""}.</h1>'
                f'<p class="lede">{now.strftime("%A %d %B").replace(" 0", " ")} · {fact}</p>',
                unsafe_allow_html=True
            )
        with action:
            if st.button("Find my matches", key="home_try", type="primary", icon=":material/arrow_forward:",
                         use_container_width=True):
                set_page("Dashboard")

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # Each card is a Streamlit container, so it can hold real buttons. The
    # graphics are plain HTML that animate once the card scrolls into view.
    row1 = st.columns([2, 1], gap="medium")
    with row1[0], st.container(key="bento_skills"):
        bento_open("skills", "Profile", "Your CV")
        if saved_name:
            title = f"{len(found)} of {len(MATCH_SKILLS)} skills found in your CV"
            sub = "Blue ones the matcher found. Dashed ones it looks for but couldn't see yet."
        else:
            title = "Upload a CV to see your skills"
            sub = f"The matcher reads {len(MATCH_SKILLS)} skills. These are the ones it looks for."
        cloud = "".join(f'<span class="chip chip-have" style="--i:{i}">{escape(s)}</span>'
                        for i, s in enumerate(found))
        cloud += "".join(f'<span class="chip chip-missing" style="--i:{len(found) + i}">{escape(s)}</span>'
                         for i, s in enumerate(not_found))
        st.markdown(bento_head(title, sub, "sparkles", "blue") + f'<div class="bento-graphic cloud">{cloud}</div>',
                    unsafe_allow_html=True)

    with row1[1], st.container(key="bento_best"):
        bento_open("best", "Dashboard", "Matches")
        if last:
            off = 339.3 * (1 - last["score"] / 100)
            ring = (f'<div class="ring"><svg viewBox="0 0 120 120"><circle class="track" cx="60" cy="60" r="54"/>'
                    f'<circle class="arc" cx="60" cy="60" r="54" style="--off:{off:.1f}"/></svg>'
                    f'<div class="ring-num num"><span class="count whole-only" style="--to-whole:{int(last["score"])};'
                    f'--to-tenth:0"></span>%</div></div>')
            meta = (f'<div class="ring-meta"><b>{escape(last["title"])}</b>'
                    f'<span>{escape(last["company"])}</span></div>')
            st.markdown(bento_head("Your best match", icon_name="trophy", color="gold") +
                        f'<div class="bento-graphic ring-wrap">{ring}{meta}</div>', unsafe_allow_html=True)
        else:
            # No match yet: a CV waiting inside the empty score ring.
            ring = ('<div class="ring ring-empty"><svg viewBox="0 0 120 120"><circle class="track" cx="60" cy="60" r="54"/>'
                    f'</svg><div class="ring-num">{icon("cv", "orange")}</div></div>')
            sub = ("Run a match and your top role lands here." if saved_name
                   else "Upload your CV, run a match, and your top role lands here.")
            st.markdown(bento_head("Your best match", sub, "trophy", "gold") +
                        f'<div class="bento-graphic ring-wrap">{ring}</div>', unsafe_allow_html=True)
            if st.button("Run my first match", key="home_first_match"):
                set_page("Dashboard")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    row2 = st.columns(3, gap="medium")
    with row2[0], st.container(key="bento_how"):
        bento_open("how", "About", "How it works")
        bars = (
            '<div class="wbar"><div class="wbar-top"><b>Skills you have</b><span class="num">75%</span></div>'
            '<div class="wbar-track"><div class="wbar-fill skills" style="width:75%"></div></div></div>'
            '<div class="wbar"><div class="wbar-top"><b>Wording match</b><span class="num">25%</span></div>'
            '<div class="wbar-track"><div class="wbar-fill words" style="width:25%"></div></div></div>'
            '<div class="lifts"><span class="lift" style="--i:0">+12 field</span>'
            '<span class="lift" style="--i:1">+10 location</span><span class="lift" style="--i:2">+8 job type</span></div>'
        )
        st.markdown(bento_head("How a score is made", "Then your preferences lift the roles that fit them.", "bars", "green") +
                    f'<div class="bento-graphic">{bars}</div>', unsafe_allow_html=True)

    with row2[1], st.container(key="bento_cv"):
        bento_open("cv", "Profile", "Your CV")
        # A page of the CV: line widths vary like text, and the share of
        # highlighted lines follows the share of skills found.
        widths = [92, 78, 85, 60, 88, 72, 95, 66, 80, 54]
        hits = round(len(widths) * len(found) / max(len(MATCH_SKILLS), 1)) if saved_name else 0
        hit_rows = set(range(1, len(widths), 2)[:hits]) | set(range(0, len(widths), 2)[:max(0, hits - 5)])
        lines = "".join(f'<i class="{"hit" if i in hit_rows else ""}" style="width:{w}%;--i:{i}"></i>'
                        for i, w in enumerate(widths))
        name = escape(saved_name) if saved_name else "No CV yet"
        st.markdown(bento_head("Your CV, as the matcher reads it", icon_name="file", color="orange") +
                    f'<div class="bento-graphic"><div class="paper">{lines}<div class="scan"></div></div>'
                    f'<div class="paper-name">{name}</div></div>', unsafe_allow_html=True)
        if st.button("Manage CV" if saved_name else "Upload a CV", key="home_profile"):
            set_page("Profile" if saved_name else "Dashboard")

    with row2[2], st.container(key="bento_prefs"):
        bento_open("prefs", "Profile", "Preferences")
        prefs = [("Location", profile["preferred_location"]), ("Job type", profile["job_type"]),
                 ("Field", profile["field_of_interest"])]
        rows = "".join(f'<div class="pref"><span>{k}</span><b class="{"" if v else "any"}">{escape(v or "Any")}</b></div>'
                       for k, v in prefs)
        st.markdown(bento_head("What lifts a role for you", icon_name="sliders", color="pink") + f'<div class="bento-graphic">{rows}</div>',
                    unsafe_allow_html=True)
        if st.button("Edit preferences", key="home_prefs"):
            set_page("Profile")


def winner_sparks(count=28):
    """
    Gold sparks for the best-match card's winner moment. Each spark gets its own
    direction, distance, size and duration, so they fly out together and then
    stop one by one. Values come from the spark's index, not random, so the
    burst looks the same every time.
    """
    sparks = []
    for k in range(count):
        angle = 2 * math.pi * k / count + (k * 0.37) % 0.3
        reach = 170 + (k * 53) % 150
        # The card is wide and short, so the burst is an ellipse to match.
        x, y = math.cos(angle) * reach * 1.7, math.sin(angle) * reach * 0.75
        duration = 1100 + (k * 97) % 1500
        size = 6 + k % 5
        sparks.append(f'<i style="--x:{x:.0f}px;--y:{y:.0f}px;--d:{duration}ms;--s:{size}px"></i>')
    return f'<div class="sparks" aria-hidden="true">{"".join(sparks)}</div>'


STEP_MIN_SECONDS = 0.8
ANALYSIS_STEPS = ["Reading your CV", "Checking GitHub for more skills", "Shortlisting and scoring roles",
                  "Picking your best match"]


def show_steps(slot, current, roles=None):
    """
    Draw the analysis steps into `slot`: finished ones ticked, the current one
    pulsing, the rest waiting. Once every step is done (current == 4) it shrinks
    to one quiet line that stays above the results.
    """
    # Each step stays up at least STEP_MIN_SECONDS so it can be read; the real
    # work is often faster than that and would otherwise flash past.
    shown_at = st.session_state.get("_step_shown_at")
    if current > 0 and shown_at is not None:
        wait = STEP_MIN_SECONDS - (time.monotonic() - shown_at)
        if wait > 0:
            time.sleep(wait)
    st.session_state["_step_shown_at"] = time.monotonic()
    if current >= len(ANALYSIS_STEPS):
        done = ["Read your CV", f"scored {roles} roles" if roles else "scored the roles", "picked the best"]
        slot.markdown('<div class="steps-done">' + " · ".join(f'<span>✓ {d}</span>' for d in done) + '</div>',
                      unsafe_allow_html=True)
        return
    items = ""
    for i, label in enumerate(ANALYSIS_STEPS):
        state = "done" if i < current else "now" if i == current else ""
        items += f'<div class="step {state}"><span class="dot">{"✓" if state == "done" else ""}</span>{label}</div>'
    slot.markdown(f'<div class="steps">{items}</div>', unsafe_allow_html=True)


def whatif_html(results):
    """
    The what-if simulator: pick one of the top roles, tap the skills it's missing,
    and watch the score and bar update instantly. It runs in the browser (an HTML
    component), so it reacts without Streamlit reloading the page. The maths is
    the app's own: 0.75 x skill share + 0.25 x wording similarity.
    """
    roles = [{
        "title": r["title"], "company": r["company"],
        "have": r["matched_skills"], "missing": r["missing_skills"],
        "skill": r["skill_score"], "sim": r["similarity_score"],
    } for r in results]
    return """
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root { THEME_TOKENS }
  * { box-sizing: border-box; }
  /* No scrollbar: the frame is sized to fit this content (see below). */
  html, body { background: transparent; overflow: hidden; }
  body { margin: 0; font-family: "Instrument Sans", system-ui, sans-serif; color: var(--text); }
  .wrap { background-color: var(--surface);
          background-image: radial-gradient(420px 260px at 0 0, color-mix(in srgb, var(--c-green) var(--panel-glow), transparent), transparent 70%),
                            var(--grain), linear-gradient(180deg, var(--surface-top) 0%, var(--surface) 62%);
          border: 1px solid var(--edge); box-shadow: inset 0 1px 0 var(--edge-hi);
          border-radius: 16px; padding: 20px; }
  .title svg { width: calc(16px + 1.5 * var(--icon-pad)); height: calc(16px + 1.5 * var(--icon-pad));
               margin-right: 7px; border-radius: 8px;
               padding: calc(var(--icon-pad) * .75); vertical-align: calc(-3px - var(--icon-pad) * .75);
               background: color-mix(in srgb, currentColor var(--icon-tint), transparent); }
  .head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
  .title { font-size: 14px; font-weight: 600; }
  .sub { font-size: 12px; color: var(--muted); }
  .roles { display: flex; gap: 6px; flex-wrap: wrap; margin: 14px 0 16px; }
  .role { font: inherit; font-size: 13px; padding: 6px 12px; border-radius: 999px; cursor: pointer;
          border: 1px solid var(--control); background: transparent; color: var(--text);
          transition: background-color 150ms ease, color 150ms ease, transform 120ms cubic-bezier(0.23, 1, 0.32, 1); }
  .role:active, .chip:active { transform: scale(0.96); }
  .role.on { background: var(--primary-bg); border-color: var(--primary-bg); color: var(--primary-fg); }
  .row { display: flex; justify-content: space-between; align-items: baseline; }
  .score { font-size: 32px; font-variant-numeric: tabular-nums; }
  .gain { font-size: 13px; color: var(--accent); font-weight: 500; min-height: 18px; }
  .track { display: flex; gap: 2px; height: 10px; border-radius: 999px; background: var(--track); overflow: hidden; margin: 10px 0 8px; }
  .sk { background: var(--accent); transition: width 600ms cubic-bezier(0.33, 1, 0.68, 1); }
  .wd { background: var(--accent-soft); transition: width 600ms cubic-bezier(0.33, 1, 0.68, 1); }
  .note { font-size: 13px; color: var(--muted); }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
  .chip { font: inherit; font-size: 12px; font-weight: 500; padding: 4px 11px; border-radius: 999px; cursor: pointer;
          border: 1px dashed var(--control); background: transparent; color: var(--text);
          transition: background-color 150ms ease, border-color 150ms ease, transform 120ms cubic-bezier(0.23, 1, 0.32, 1); }
  .chip.on { background: var(--accent-tint); border: 1px solid var(--accent-soft); color: var(--accent-ink); }
  .chip.have { background: var(--chip-muted-bg); border: 1px solid transparent; color: var(--muted); cursor: default; }
  .empty { font-size: 13px; color: var(--muted); margin-top: 12px; }
  @media (prefers-reduced-motion: reduce) { .sk, .wd { transition: none; } }
</style>
<div class="wrap">
  <div class="head"><div class="title">ZAP_ICONWhat if you learned a skill?</div><div class="sub">Tap a missing skill to try it</div></div>
  <div class="roles" id="roles"></div>
  <div class="row"><div class="score" id="score"></div><div class="gain" id="gain"></div></div>
  <div class="track"><div class="sk" id="sk"></div><div class="wd" id="wd"></div></div>
  <div class="note" id="note"></div>
  <div class="chips" id="chips"></div>
</div>
<script>
const roles = ROLES_JSON;
let pick = 0, added = new Set();
const r1 = v => Math.round(v * 10) / 10;
function scoreOf(role, extra) {
  const req = role.have.length + role.missing.length, n = role.have.length + extra;
  if (n === 0) return { skills: 0, words: r1(role.sim) };
  const share = extra === 0 ? role.skill : n / req * 100;
  return { skills: r1(0.75 * share), words: r1(0.25 * role.sim) };
}
function draw() {
  const role = roles[pick], base = scoreOf(role, 0), now = scoreOf(role, added.size);
  const total = r1(now.skills + now.words), baseTotal = r1(base.skills + base.words);
  document.getElementById("score").textContent = total.toFixed(1) + "%";
  document.getElementById("gain").textContent = added.size ? "+" + (total - baseTotal).toFixed(1) + " points" : "";
  document.getElementById("sk").style.width = now.skills + "%";
  document.getElementById("wd").style.width = now.words + "%";
  const req = role.have.length + role.missing.length;
  document.getElementById("note").textContent =
    `You have ${role.have.length + added.size} of the ${req} skills ${role.title} lists.`;
  const chips = document.getElementById("chips");
  chips.innerHTML = "";
  if (!role.missing.length) {
    chips.innerHTML = '<div class="empty">You already have every skill this role lists. The wording match decides the rest.</div>';
  }
  role.missing.forEach(s => {
    const b = document.createElement("button");
    b.className = "chip" + (added.has(s) ? " on" : "");
    b.textContent = (added.has(s) ? "✓ " : "+ ") + s;
    b.setAttribute("aria-pressed", added.has(s));
    b.onclick = () => { added.has(s) ? added.delete(s) : added.add(s); draw(); };
    chips.appendChild(b);
  });
  role.have.forEach(s => {
    const c = document.createElement("span"); c.className = "chip have"; c.textContent = s; chips.appendChild(c);
  });
}
const rolesEl = document.getElementById("roles");
roles.forEach((role, i) => {
  const b = document.createElement("button");
  b.className = "role" + (i === pick ? " on" : "");
  b.textContent = role.title;
  b.onclick = () => {
    pick = i; added = new Set();
    [...rolesEl.children].forEach((c, j) => c.classList.toggle("on", j === i));
    draw();
  };
  rolesEl.appendChild(b);
});
// Start on the first role that has something to try.
const firstGap = roles.findIndex(r => r.missing.length);
if (firstGap > 0) rolesEl.children[firstGap].click(); else draw();
</script>
""".replace("ROLES_JSON", json.dumps(roles).replace("</", "<\\/")).replace(
        "THEME_TOKENS", ";".join(f"--{k}:{v}" for k, v in THEMES[st.session_state.theme].items())).replace(
        "ZAP_ICON", icon("zap", "green"))


def score_breakdown(result):
    """
    The final score taken apart: 0.75 x skill overlap plus 0.25 x wording
    similarity (match_agent.calculate_final_score). The figure shown is the sum of
    the two parts as displayed, so the parts always add up to it.
    """
    matched = len(result["matched_skills"])
    required = matched + len(result["missing_skills"])
    if result["skill_score"] == 0:
        # No listed skills detected in the job: the score is wording alone.
        words = round(result["similarity_score"], 1)
        return words, f"""
        <div class="bd"><div class="bd-track"><div class="bd-seg words" style="width:{words}%"></div></div>
        <div class="bd-legend"><div class="bd-item"><span class="sw words"></span>
          <b class="num">{words:.1f} points from wording</b>No required skills were found in this job's text,
          so the score is wording similarity alone.</div></div></div>"""
    skills_pts = round(0.75 * result["skill_score"], 1)
    words_pts = round(0.25 * result["similarity_score"], 1)
    total = round(skills_pts + words_pts, 1)
    return total, f"""
    <div class="bd"><div class="bd-track">
        <div class="bd-seg skills" style="width:{skills_pts}%"></div>
        <div class="bd-seg words" style="width:{words_pts}%"></div></div>
      <div class="bd-legend num">
        <div class="bd-item"><b><span class="sw skills"></span>{skills_pts:.1f} points from skills</b>
          You have {matched} of the {required} skills this role lists.</div>
        <div class="bd-item"><b><span class="sw words"></span>{words_pts:.1f} points from wording</b>
          Your CV and the job text are {result['similarity_score']:.0f}% alike.</div>
      </div></div>"""


def page_dashboard():
    with st.container(key="hero_matches"):
        head, action = st.columns([9, 3], vertical_alignment="bottom")
        with head:
            title_slot = st.empty()
        with action:
            run = st.button("Analyze my CV", use_container_width=True, key="analyze_btn", type="primary",
                            icon=":material/play_arrow:")
            # Saved preferences change the ranking, so an existing analysis runs again with them.
            run = st.session_state.pop("rerank", False) or run

    def title(sentence):
        title_slot.markdown(f'<h1 class="page-title">Matches</h1><p class="lede">{sentence}</p>',
                            unsafe_allow_html=True)

    title("Rank the open roles against your CV, and see which skills each one is missing.")
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    cv_data, cv_problem = None, None
    if run:
        if st.session_state.cv_name:
            cv_data, cv_problem = read_cv(st.session_state.cv_name, st.session_state.cv_bytes or b"")
        else:
            cv_problem = "Upload a CV first."

    left, right = st.columns([4, 8], gap="small")

    with left:
        with st.container(key="panel_upload"):
            st.markdown(f'<div class="panel-head">{panel_title("Your CV", "file", "orange")}</div>', unsafe_allow_html=True)

            up = st.file_uploader(
                "Upload a CV (PDF, DOCX or TXT)",
                type=["pdf", "docx", "txt"],
                key="cv_uploader",
                label_visibility="collapsed"
            )

            # The uploader keeps its file across reruns; only a new file counts as a new CV.
            if up is not None and up.file_id != st.session_state.get("cv_file_id"):
                st.session_state.cv_file_id = up.file_id
                cv_bytes = up.read()
                save_user_cv(up.name, cv_bytes)
                cv_problem = read_cv(up.name, cv_bytes)[1]
                if not cv_problem:
                    st.toast("CV ready to match.", icon=":material/check:")

            # Shown here, not with the results: on a phone those are far below the Analyze button.
            if cv_problem:
                st.error(cv_problem)

            if st.session_state.cv_name:
                st.markdown(
                    f'<div class="row-card" style="margin:4px 0 12px"><div class="row-title" '
                    f'style="overflow:hidden;text-overflow:ellipsis">{escape(st.session_state.cv_name)}</div>'
                    f'<div class="row-meta">Kept for this visit</div></div>',
                    unsafe_allow_html=True
                )


        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Quick Preferences ──
        with st.container(key="panel_prefs"):
            st.markdown(f'<div class="panel-head">{panel_title("Preferences", "sliders", "pink")}'
                        '<span>Lift matching roles</span></div>', unsafe_allow_html=True)

            dash_prefs = load_user_profile()

            loc_index = LOCATION_OPTIONS.index(dash_prefs["preferred_location"]) if dash_prefs["preferred_location"] in LOCATION_OPTIONS else 0
            jt_index = JOB_TYPE_OPTIONS.index(dash_prefs["job_type"]) if dash_prefs["job_type"] in JOB_TYPE_OPTIONS else 0
            fi_index = FIELD_OPTIONS.index(dash_prefs["field_of_interest"]) if dash_prefs["field_of_interest"] in FIELD_OPTIONS else 0

            dash_location = st.selectbox("Location", LOCATION_OPTIONS, index=loc_index,
                key="dash_location", format_func=lambda x: "Any" if x == "" else x)
            dash_job_type = st.selectbox("Job type", JOB_TYPE_OPTIONS, index=jt_index,
                key="dash_job_type", format_func=lambda x: "Any" if x == "" else x)
            dash_field = st.selectbox("Field", FIELD_OPTIONS, index=fi_index,
                key="dash_field", format_func=lambda x: "Any" if x == "" else x)

            if st.button("Save preferences", key="dash_save_prefs", use_container_width=True):
                save_user_profile(dash_prefs["full_name"], dash_location, dash_job_type, dash_field, dash_prefs.get("github_username", ""))
                st.toast("Preferences saved.", icon=":material/check:")
                st.rerun()

    with right:
        with st.container(key="panel_results"):
            if run:
                if not cv_problem:
                    # The analysis shows its real steps as they happen, in place of a spinner.
                    steps_slot = st.empty()
                    show_steps(steps_slot, current=0)

                    # ── Enrich skills with GitHub ──
                    github_skills = []
                    github_username = load_user_profile()["github_username"]
                    if github_username.strip():
                        show_steps(steps_slot, current=1)
                        github_skills = fetch_github_skills(github_username)
                        if github_skills:
                            # Merge GitHub skills into CV skills
                            merged_skills = list(set(cv_data["skills"] + github_skills))
                            cv_data["skills"] = merged_skills
                            # Also append to text for TF-IDF matching
                            cv_data["text"] += " " + " ".join(github_skills)
                            st.toast(f"Added {len(github_skills)} skills from GitHub.", icon=":material/check:")

                    show_steps(steps_slot, current=2)
                    jobs = load_jobs()
                    match_results = analyze_all_jobs(cv_data, jobs)
                    show_steps(steps_slot, current=3)

                    # ── Apply preference boosts ──
                    prefs = load_user_profile()
                    pref_location = prefs.get("preferred_location", "").lower().strip()
                    pref_job_type = prefs.get("job_type", "").lower().strip()
                    pref_field    = prefs.get("field_of_interest", "").lower().strip()

                    FIELD_KEYWORDS = {
                        "artificial intelligence": ["ai", "artificial intelligence", "machine learning", "nlp", "deep learning"],
                        "web development": ["web", "frontend", "backend", "html", "css", "javascript", "react"],
                        "data science": ["data", "analyst", "analytics", "sql", "python", "visualization"],
                        "cybersecurity": ["security", "cyber", "penetration", "firewall", "soc"],
                        "mobile development": ["mobile", "android", "ios", "flutter", "swift"],
                        "cloud computing": ["cloud", "aws", "azure", "devops", "kubernetes"],
                        "software engineering": ["software", "developer", "engineer", "programming"],
                        "devops": ["devops", "ci/cd", "docker", "kubernetes", "pipeline"],
                    }

                    def preference_boost(result):
                        score = result["final_score"]
                        job_text = (result.get("title","") + " " + result.get("description","") + " " + result.get("location","")).lower()

                        # Location boost: +10 if matches
                        if pref_location and pref_location not in ["any", "remote", ""]:
                            if pref_location in job_text:
                                score += 10

                        # Job type boost: +8 if matches
                        if pref_job_type and pref_job_type not in [""]:
                            if pref_job_type in job_text:
                                score += 8

                        # Field boost: +12 if keyword matches
                        if pref_field:
                            keywords = FIELD_KEYWORDS.get(pref_field.lower(), [pref_field])
                            # Whole words only: "ai" must not match "email" or "training".
                            if any(re.search(rf"\b{re.escape(kw)}\b", job_text) for kw in keywords):
                                score += 12

                        return min(score, 100)  # cap at 100

                    if pref_location or pref_job_type or pref_field:
                        for r in match_results:
                            r["boosted_score"] = preference_boost(r)
                        match_results.sort(key=lambda x: x["boosted_score"], reverse=True)
                    else:
                        for r in match_results:
                            r["boosted_score"] = r["final_score"]

                    recommendations = generate_recommendations(match_results, top_n=len(match_results))
                    show_steps(steps_slot, current=4, roles=len(match_results))

                    # What shaped the ranking, as one quiet line.
                    shaped = []
                    if pref_location and pref_location != "any":
                        shaped.append(prefs["preferred_location"])
                    if pref_job_type:
                        shaped.append(prefs["job_type"])
                    if pref_field:
                        shaped.append(prefs["field_of_interest"])

                    # Kept for this visit, so the results stay on screen when anything else
                    # on the page changes. A new run number means Streamlit builds the result
                    # blocks fresh, so the reveal animation plays for each new analysis.
                    st.session_state.analysis = {
                        "cv_name": st.session_state.cv_name, "match_results": match_results,
                        "recommendations": recommendations, "shaped": shaped,
                        "github_added": len(github_skills) if github_username.strip() else 0}
                    st.session_state.analysis_runs = st.session_state.get("analysis_runs", 0) + 1
                    if match_results:
                        save_best_match(match_results[0]["title"], match_results[0]["company"],
                                        score_breakdown(match_results[0])[0])

            # The latest analysis of this CV, whether it ran just now or earlier in the visit.
            analysis = load_analysis()
            if analysis and analysis["match_results"]:
                match_results = analysis["match_results"]
                recommendations = analysis["recommendations"]
                shaped = analysis["shaped"]
                run_no = st.session_state.analysis_runs
                top_result = match_results[0]
                total, breakdown_html = score_breakdown(top_result)

                verdict = "ranks first with your preferences" if shaped else "fits best"
                finding = (f"<strong>{escape(top_result['title'])}</strong> at {escape(top_result['company'])} "
                           f"{verdict}, at <strong>{total:.1f}%</strong>.")
                if is_senior(top_result):
                    finding += " It's a senior role, so check the experience it asks for."
                if top_result["missing_skills"]:
                    finding += f" Adding {escape(top_result['missing_skills'][0])} to your CV would lift it most."
                title(finding)

                meta_bits = [f"{len(match_results)} roles shortlisted"]
                if shaped:
                    meta_bits.append("ranked with your preferences: " + ", ".join(escape(s) for s in shaped))
                if analysis["github_added"]:
                    meta_bits.append(f'{analysis["github_added"]} skills added from GitHub')
                st.markdown(
                    f'<div class="panel-head">{panel_title("Best match", "trophy", "gold")}'
                    f'<span>{" · ".join(meta_bits)}</span></div>',
                    unsafe_allow_html=True
                )

                best = recommendations[0] if recommendations else None
                fit = best["fit_label"] if best else ""
                missing_html = (chips(top_result["missing_skills"], "missing")
                                or '<span class="muted">None. You cover every listed skill.</span>')
                matched_html = (chips(top_result["matched_skills"], "have")
                                or '<span class="muted">None of the listed skills yet.</span>')

                if is_example(top_result):
                    notice = ("This is an example role written for the demo, not a real opening. "
                              "The link searches for similar real jobs.")
                else:
                    notice = "A live job from Remote OK. AI Match only scores the fit; to apply, open the listing."

                with st.container(key=f"card_top_{run_no}"):
                    st.markdown(
                        f"""
                        {winner_sparks()}
                        <div class="top-head">
                          <div><div class="top-title">{escape(top_result['title'])}</div>
                            <div class="top-meta">{escape(top_result['company'])} · {escape(top_result['location'])}{source_tag(top_result)}{seniority_tag(top_result)}{preference_tag(top_result)}</div>
                            <div style="margin-top:10px"><span class="chip chip-fit {'strong' if fit == 'Strong Match' else ''}">{escape(fit)}</span>
                              <span class="top-pick"><svg width="13" height="13" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4z"/><path d="M17 5h3v2a3 3 0 0 1-3 3M7 5H4v2a3 3 0 0 0 3 3"/></svg>Ranked #1 of {len(match_results)} roles</span></div>
                          </div>
                          <div class="top-score num" aria-label="{total:.1f}% match score">
                            <span class="count" aria-hidden="true" style="--to-whole:{int(total)};--to-tenth:{int(round(total * 10)) % 10}"></span><span aria-hidden="true">%</span>
                            <small>match score</small></div>
                        </div>
                        {breakdown_html}
                        <div class="skill-block"><div class="label">You have</div>{matched_html}</div>
                        <div class="skill-block"><div class="label">Missing</div>{missing_html}</div>
                        <div class="notice">{notice}
                          <a class="link" style="float:right" href="{escape(top_result['url'])}" target="_blank">{job_link_label(top_result)}</a></div>
                        """,
                        unsafe_allow_html=True
                    )

                with st.container(key=f"whatif_{run_no}"):
                    # "content" sizes the frame to the simulator itself, so it never
                    # needs a scrollbar however many rows the skill chips wrap to.
                    st.iframe(whatif_html(match_results[:3]), height="content")

                if len(recommendations) > 1:
                    st.markdown(f'<div class="panel-head" style="margin-top:24px">{panel_title("Also worth a look", "compass", "blue")}</div>', unsafe_allow_html=True)
                    rows = ""
                    # Recommendations are the first match results in the same
                    # order, but carry only the final score; the breakdown needs
                    # the full result.
                    for rec, result in zip(recommendations[1:], match_results[1:]):
                        rec_total, _ = score_breakdown(result)
                        gaps = chips(rec["missing_skills"][:5], "missing") or '<span class="muted">No missing skills.</span>'
                        # Shown on hover: where this role lands if every gap is closed.
                        if result["missing_skills"]:
                            ceiling = round(75.0 + round(0.25 * result["similarity_score"], 1), 1)
                            names = ", ".join(escape(s) for s in result["missing_skills"][:3])
                            more = (f"Add {names} to your CV and this role reaches "
                                    f"<strong class='num'>{ceiling:.1f}%</strong>.")
                        else:
                            more = "You cover every skill it lists; only the wording match holds it back."
                        # The card takes the colour of its fit: green strong, blue moderate, pink weak.
                        fit_colour = {"Strong Match": "green", "Moderate Match": "blue"}.get(rec["fit_label"], "pink")
                        rows += f"""
                        <div class="row-card" style="--wc1:var(--c-{fit_colour})">
                          <div class="row-top">
                            <div><div class="row-title">{escape(rec['title'])}</div>
                              <div class="row-meta">{escape(rec['company'])} · {escape(rec['location'])} ·
                                {len(rec['matched_skills'])} of {len(rec['matched_skills']) + len(rec['missing_skills'])} skills{source_tag(result)}{seniority_tag(result)}{preference_tag(result)}</div></div>
                            <div class="row-score num">{rec_total:.1f}%<span class="row-fit">{escape(rec['fit_label'])}</span></div>
                          </div>
                          <div class="row-body">{gaps}</div>
                          <div class="row-more"><div>{more}</div></div>
                          <a href="{escape(rec['url'])}" target="_blank">{job_link_label(result)}</a>
                        </div>"""
                    st.container(key=f"others_{run_no}").markdown(
                        f'<div class="rows rows-results">{rows}</div>', unsafe_allow_html=True)

            elif analysis:
                st.warning("No jobs were available for matching.")
            else:
                st.markdown(f'<div class="panel-head">{panel_title("Best match", "trophy", "gold")}</div>'
                            '<div class="empty">Nothing ranked yet. Press <strong>Analyze my CV</strong> and the best '
                            'role appears here, with its score taken apart and the skills it is missing.</div>',
                            unsafe_allow_html=True)


def page_about():
    with st.container(key="hero_about"):
        st.markdown('<h1 class="page-title">How AI Match works</h1>'
                    '<p class="lede">An NLP recruitment tool, built as a graduation project for the BSc in Computer '
                    'Science (Artificial Intelligence) at King Khalid University.</p>',
                    unsafe_allow_html=True)
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    left, right = st.columns([8, 4], gap="large")
    with left:
        st.markdown(
            """
            <div class="prose">
            <p>AI Match reads a CV, compares it with job descriptions, scores how well each fits,
            and tells the candidate which skills they are missing, as feedback rather than a pass or fail.</p>

            <h2>Two stages</h2>
            <p><strong>Retrieval.</strong> Every job description is turned into an embedding with
            sentence-transformers (all-MiniLM-L6-v2) and indexed in FAISS. The CV is embedded the same way,
            and FAISS returns the five nearest jobs. This stage recognises related wording, not just exact keywords.
            Two of the five spots are saved for the live jobs that score best, because real posts rarely read as
            closely to a CV as the example roles do.</p>
            <p><strong>Ranking.</strong> Each shortlisted job is then scored two ways:</p>
            <table>
              <tr><th>Part</th><th class="num">Weight</th><th>What it measures</th></tr>
              <tr><td>Skill overlap</td><td class="num">75%</td><td>How many of the job's required skills appear in the CV</td></tr>
              <tr><td>TF-IDF similarity</td><td class="num">25%</td><td>How similar the two documents read as text</td></tr>
            </table>
            <p>Semantic similarity alone gave scores too close together to be useful, so the final number
            leans on concrete skill overlap.</p>

            <h2>Five agents, one job each</h2>
            <table>
              <tr><td>CV agent</td><td>Reads the CV and detects skills</td></tr>
              <tr><td>Job agent</td><td>Fetches job listings, with sample Saudi roles as a fallback</td></tr>
              <tr><td>Vector agent</td><td>Embeds text and shortlists jobs with FAISS</td></tr>
              <tr><td>Match agent</td><td>Scores skill overlap and wording similarity</td></tr>
              <tr><td>Recommendation agent</td><td>Ranks roles and explains the gaps</td></tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        with st.container(key="panel_stack"):
            st.markdown(f'<div class="panel-head">{panel_title("Built with", "layers", "violet")}</div>', unsafe_allow_html=True)
            stack = [("Python", "The whole system"), ("Streamlit", "This interface"),
                     ("sentence-transformers", "Embeddings for retrieval"), ("FAISS", "Vector index for the shortlist"),
                     ("scikit-learn", "TF-IDF and cosine similarity"), ("pypdf, python-docx", "Reading CV files"),
                     ("Remote OK API", "Live remote jobs")]
            st.markdown('<div class="rows">' + "".join(
                f'<div class="row-card"><div class="row-title">{t}</div><div class="row-meta">{d}</div></div>'
                for t, d in stack) + '</div>', unsafe_allow_html=True)


def page_privacy():
    with st.container(key="hero_privacy"):
        st.markdown('<h1 class="page-title">Privacy</h1>'
                    '<p class="lede">What AI Match keeps about you, and how to have it removed.</p>',
                    unsafe_allow_html=True)
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    left, _ = st.columns([8, 4], gap="large")
    left.markdown(PRIVACY_HTML, unsafe_allow_html=True)


def page_contact():
    with st.container(key="hero_contact"):
        st.markdown('<h1 class="page-title">Contact</h1>'
                    '<p class="lede">AI Match is a student project by Faisal Alhudaithy. Questions, feedback and bug '
                    'reports are welcome.</p>',
                    unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="small")
    links = [
        (c1, "Report a bug or suggest a feature", "Open an issue on the project's GitHub page.",
         "https://github.com/faisal5619/ai-match/issues", "Open GitHub issues", "bug", "pink"),
        (c2, "Read the code", "The full source, with a README on how the matching works.",
         "https://github.com/faisal5619/ai-match", "View the repository", "code", "green"),
        (c3, "Get in touch", "For anything else, message Faisal on LinkedIn.",
         LINKEDIN_URL, "Open LinkedIn", "chat", "blue"),
    ]
    for i, (col, t, d, href, cta, ic, colour) in enumerate(links):
        with col:
            with st.container(key=f"panel_contact_{i}"):
                st.markdown(
                    f'{panel_title(t, ic, colour)}<p class="muted" style="margin:6px 0 16px">{d}</p>'
                    f'<a class="link" href="{href}" target="_blank">{cta}</a>',
                    unsafe_allow_html=True
                )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    with st.container(key="panel_project"):
        st.markdown(f'<div class="panel-head">{panel_title("About the project", "info", "blue")}</div>',
                    unsafe_allow_html=True)
        a, b = st.columns(2, gap="large")
        facts_a = [("Author", "Faisal Alhudaithy"),
                   ("Degree", "BSc Computer Science, Artificial Intelligence"),
                   ("University", "King Khalid University")]
        facts_b = [("Job data", "Example Saudi roles, plus live jobs from Remote OK when reachable"),
                   ("Hosting", "Streamlit Community Cloud, free tier"),
                   ("If it's asleep", "The free tier sleeps after 12 hours idle; waking takes about 30 seconds")]
        for col, facts in ((a, facts_a), (b, facts_b)):
            col.markdown("".join(f'<div class="kv"><span>{k}</span><span style="white-space:normal">{v}</span></div>'
                                 for k, v in facts), unsafe_allow_html=True)


def page_profile():
    with st.container(key="hero_profile"):
        st.markdown('<h1 class="page-title">Profile</h1>'
                    '<p class="lede">Your details, the preferences that lift matching roles, and the CV you match with.</p>',
                    unsafe_allow_html=True)
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    profile = load_user_profile()
    saved_name, saved_bytes = load_user_cv()

    # ── Details and preferences: one panel, one Save ──
    label, panel = st.columns([5, 7], gap="large")
    with label:
        gh = profile.get("github_username", "")
        summary = [
            ("Name", profile["full_name"] or "—"),
            ("GitHub", f"@{gh}" if gh else "—"),
            ("Location", profile["preferred_location"] or "Any"),
            ("Job type", profile["job_type"] or "Any"),
            ("Field", profile["field_of_interest"] or "Any"),
        ]
        st.markdown(
            set_label_head("Details and preferences", "user", "green") +
            '<p>Preferences lift roles that fit them: location by 10 points, job type by 8 and field by 12. '
            'A GitHub username adds the languages in your public repositories to your skills.</p></div>'
            '<div style="margin-top:16px">' + "".join(
                f'<div class="kv"><span>{k}</span><span>{escape(v)}</span></div>' for k, v in summary) + '</div>',
            unsafe_allow_html=True
        )
    with panel:
        with st.container(key="panel_details"):
            new_name  = st.text_input("Full name", value=profile["full_name"], key="prof_name")
            new_github = st.text_input("GitHub username", value=profile.get("github_username", ""),
                key="prof_github", placeholder="e.g. faisal5619")

            loc_index = LOCATION_OPTIONS.index(profile["preferred_location"]) if profile["preferred_location"] in LOCATION_OPTIONS else 0
            new_location = st.selectbox(
                "Preferred location",
                options=LOCATION_OPTIONS,
                index=loc_index,
                key="prof_location",
                format_func=lambda x: "Any" if x == "" else x
            )

            jt_index = JOB_TYPE_OPTIONS.index(profile["job_type"]) if profile["job_type"] in JOB_TYPE_OPTIONS else 0
            new_job_type = st.selectbox(
                "Job type",
                options=JOB_TYPE_OPTIONS,
                index=jt_index,
                key="prof_job_type",
                format_func=lambda x: "Any" if x == "" else x
            )

            fi_index = FIELD_OPTIONS.index(profile["field_of_interest"]) if profile["field_of_interest"] in FIELD_OPTIONS else 0
            new_field = st.selectbox(
                "Field of interest",
                options=FIELD_OPTIONS,
                index=fi_index,
                key="prof_field",
                format_func=lambda x: "Any" if x == "" else x
            )

            if st.button("Save profile", key="save_profile_btn", type="primary"):
                save_user_profile(new_name, new_location, new_job_type, new_field, new_github)
                st.toast("Profile saved.", icon=":material/check:")
                st.rerun()

    st.markdown('<div class="set-rule"></div>', unsafe_allow_html=True)

    # ── CV ──
    label, panel = st.columns([5, 7], gap="large")
    with label:
        st.markdown(
            set_label_head("CV", "file", "orange") +
            '<p>The CV every match runs against. PDF, Word or plain text. It stays for this visit only.</p></div>',
            unsafe_allow_html=True
        )
    with panel:
        with st.container(key="panel_cvfile"):
            if saved_name:
                st.markdown(
                    f'<div class="row-card" style="margin-bottom:12px"><div class="row-title" '
                    f'style="overflow:hidden;text-overflow:ellipsis">{escape(saved_name)}</div>'
                    f'<div class="row-meta">Kept for this visit</div></div>',
                    unsafe_allow_html=True
                )
                new_cv = st.file_uploader("Replace CV", type=["pdf", "docx", "txt"], key="profile_cv_upload")
                if new_cv is not None and new_cv.file_id != st.session_state.get("cv_file_id"):
                    st.session_state.cv_file_id = new_cv.file_id
                    save_user_cv(new_cv.name, new_cv.read())
                    st.toast("CV updated.", icon=":material/check:")
                    st.rerun()
            else:
                st.markdown('<div class="empty" style="margin-bottom:12px">No CV saved yet.</div>',
                            unsafe_allow_html=True)
                new_cv = st.file_uploader("Upload your CV", type=["pdf", "docx", "txt"], key="profile_cv_upload")
                if new_cv is not None and new_cv.file_id != st.session_state.get("cv_file_id"):
                    st.session_state.cv_file_id = new_cv.file_id
                    save_user_cv(new_cv.name, new_cv.read())
                    st.toast("CV uploaded.", icon=":material/check:")
                    st.rerun()

    if saved_name:
        st.markdown('<div class="set-rule"></div>', unsafe_allow_html=True)
        label, panel = st.columns([5, 7], gap="large")
        with label:
            st.markdown(
                set_label_head("Remove CV", "trash", "pink") +
                f'<p>Removes {escape(saved_name)} from this visit. Matches need a CV, so you will be asked '
                'to upload one again.</p></div>',
                unsafe_allow_html=True
            )
        with panel:
            if st.button("Delete CV", key="delete_cv_btn", icon=":material/delete:"):
                save_user_cv(None, None)
                st.toast("CV deleted.", icon=":material/delete:")
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.page not in PAGES:
    st.session_state.page = "Home"

# A short note that this is a demo, on every page until the visitor closes it.
if not st.session_state.get("note_closed"):
    with st.container(key="panel_note"):
        st.markdown(
            f'<p class="demo-note">{icon("info", "blue")}<span class="note-long"><strong>This is a demonstration project.</strong> '
            'AI Match ranks your CV against a small set of computer science roles: 20 example roles at Saudi '
            'employers, written for the demo and not real openings, plus live remote jobs from Remote OK when '
            'that site answers. Most jobs and fields are not '
            'covered yet. Your CV goes to this site\'s server, outside Saudi Arabia, and is cleared soon after you '
            'close the tab.</span>'
            # The phone version: keeps what a visitor must know, so Home's main button stays near the top.
            '<span class="note-short"><strong>Demo project.</strong> 20 example roles, not real openings, plus live '
            'remote jobs. Your CV goes to a server outside Saudi Arabia and is cleared soon after you close the '
            'tab.</span></p>',
            unsafe_allow_html=True)
        if st.button("Close this notice", key="note_close", icon=":material/close:"):
            st.session_state.note_closed = True
            st.rerun()

# Each page lives in a container keyed by its name. A new key means Streamlit
# builds the page fresh rather than reusing the last page's elements, so its
# entrance animation plays on every menu click.
with st.container(key=f"page_{st.session_state.page.lower()}"):
    if st.session_state.page == "Home":
        page_home()
    elif st.session_state.page == "Dashboard":
        page_dashboard()
    elif st.session_state.page == "Profile":
        page_profile()
    elif st.session_state.page == "About":
        page_about()
    elif st.session_state.page == "Contact":
        page_contact()
    elif st.session_state.page == "Privacy":
        page_privacy()
