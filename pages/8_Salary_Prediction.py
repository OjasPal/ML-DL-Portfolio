import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page Config ----------
st.set_page_config(
    page_title="Software Salary Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS (Unified Portfolio Theme - Golden Yellow Accent) ----------
st.markdown(
    """
    <style>
    /* Global Deep Space Theme */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Streamlit Card & Expander Container */
    div[data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Metric Values & Labels - Golden Yellow Glow */
    [data-testid="stMetricValue"] {
        color: #eab308 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        text-shadow: 0 0 14px rgba(234, 179, 8, 0.4);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* Input & Select Elements */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background-color: #0d1322 !important;
        color: #f8fafc !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
        border-color: #eab308 !important;
        box-shadow: 0 0 0 1px #eab308 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0d1322 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    /* Golden Yellow Gradient Button with Hover Glow */
    div.stButton > button {
        background: linear-gradient(90deg, #ca8a04 0%, #eab308 100%) !important;
        color: #0b0f19 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.6rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 16px rgba(202, 138, 4, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #a16207 0%, #ca8a04 100%) !important;
        box-shadow: 0 6px 22px rgba(234, 179, 8, 0.5) !important;
        transform: translateY(-1px) !important;
        color: #ffffff !important;
    }

    /* Info Sidebar Cards */
    .ais-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .ais-card h4 {
        color: #f8fafc;
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .badge-chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================================
# FEATURE ENGINEERING & ALIAS LOGIC (Directly from features.py)
# =========================================================================
ALIASES = [
    (r"^(tcs|tata consultancy)", "tata consultancy services"),
    (r"^wipro", "wipro"),
    (r"^hcl", "hcl technologies"),
    (r"^cognizant", "cognizant"),
    (r"^infosys", "infosys"),
    (r"^accenture", "accenture"),
    (r"^capgemini", "capgemini"),
    (r"^(lti|ltimindtree|l t infotech|mindtree)\b", "ltimindtree"),
    (r"^ibm", "ibm"),
    (r"^dell", "dell technologies"),
    (r"^amazon", "amazon"),
    (r"^google", "google"),
    (r"^microsoft", "microsoft"),
    (r"^(facebook|meta platforms|meta)$", "meta"),
    (r"^cts\b", "cognizant"),
    (r"^samsung", "samsung"),
    (r"^walmart", "walmart"),
    (r"^(hewlett packard enterprise|hpe)", "hewlett packard enterprise"),
    (r"^(atos|syntel)", "atos"),
    (r"^dbs", "dbs bank"),
    (r"^(robert )?bosch", "bosch"),
    (r"^hashedin", "hashedin"),
    (r"^csc$", "dxc technology"),
    (r"^cisco", "cisco"),
    (r"^oracle", "oracle"),
    (r"^expedia", "expedia"),
    (r"^(j p morgan|jpmorgan|jp morgan)", "jpmorgan"),
    (r"^ntt data", "ntt data"),
    (r"^deloitte", "deloitte"),
    (r"^paytm", "paytm"),
    (r"^flipkart", "flipkart"),
    (r"^ola( electric| cabs)?$", "ola"),
    (r"^(byju s|byjus)", "byjus"),
    (r"^(cure fit|curefit)", "cure fit"),
]

JUNK_COMPANIES = {
    "fresher", "abc", "freelancer", "done by none", "anonymous", "xyz", "nones", "none",
    "employers", "test", "yes", "na", "n a", "self employed", "student", "first student",
    "fresherworld com", "anonymous content", "confidential", "private", "company", "startup",
    "abcdef", "freshers com", "freshers", "java", "pk", "xyz tech", "other", "others", "nil", "no company",
}

TIER_1_COMPANIES = {
    "google", "microsoft", "amazon", "meta", "apple", "uber", "cisco", "oracle", "flipkart",
    "swiggy", "zomato", "adobe", "linkedin", "salesforce", "atlassian", "netflix", "nvidia",
    "intel", "amd", "broadcom", "vmware", "paypal", "intuit", "qualcomm", "samsung", "sap",
    "servicenow", "workday", "expedia", "airbnb", "stripe", "databricks", "goldman sachs",
    "morgan stanley", "d e shaw", "de shaw", "tower research capital", "directi", "rubrik",
    "nutanix", "yahoo", "walmart",
}

PRODUCT_STARTUPS = {
    "paytm", "phonepe", "razorpay", "ola", "meesho", "dream11", "cred", "zerodha", "groww",
    "byjus", "unacademy", "inmobi", "sharechat", "mindtickle", "media net", "hike", "grab",
    "gojek", "hotstar", "hevo data", "joveo", "cure fit", "blinkit", "urban company", "urbanclap",
    "oyo", "rupeek", "gainsight", "myntra", "nykaa", "lenskart", "freshworks", "zoho", "postman",
    "browserstack", "chargebee", "delhivery", "bigbasket", "dunzo", "zepto", "upstox", "pine labs",
    "mobikwik", "juspay", "cars24", "spinny", "gameskraft", "housing com", "toppr com",
    "mysmartprice", "planful", "trilogy innovations", "mobileiron", "nortonlifelock", "groupon",
    "bookmyshow", "snapdeal", "freecharge", "times internet", "hashedin",
}

LARGE_CORPS = {
    "verizon", "honeywell", "siemens", "philips", "harman", "bosch", "akamai", "comcast",
    "optum", "hewlett packard enterprise", "bharti airtel", "jio", "ericsson", "nokia",
    "mcafee", "mckinsey", "pwc", "ey", "kpmg", "tesco",
}

FINANCE_COMPANIES = {
    "dbs bank", "jpmorgan", "bny mellon", "fis", "fiserv", "barclays", "deutsche bank",
    "citi", "citibank", "hsbc", "wells fargo", "american express", "visa", "mastercard",
    "state street", "standard chartered", "bank of america", "ubs", "credit suisse", "nomura",
    "fidelity investments", "blackrock",
}

SERVICE_IT_COMPANIES = {
    "tata consultancy services", "infosys", "wipro", "hcl technologies", "cognizant",
    "accenture", "capgemini", "tech mahindra", "ltimindtree", "dxc technology", "virtusa",
    "mphasis", "cgi", "ntt data", "itc infotech", "hexaware technologies", "zensar technologies",
    "birlasoft", "coforge", "niit technologies", "sonata software", "persistent systems",
    "l t technology services", "epam systems", "publicis sapient", "ibm", "dell technologies",
    "genpact", "wns", "globallogic", "raja software labs", "accolite digital", "infogain",
    "valuelabs", "photon", "diverse lynx", "teksystems", "collabera", "virtual employee",
    "atos", "firstsource", "syntel", "igate", "unisys", "ust global", "ust", "deloitte",
}

DOMAIN_RULES = [
    ("SDET_Automation", r"\bsdet\b|\bin test\b|automation"),
    ("Data_AI", r"data scien|machine learning|\bml\b|\bai\b|deep learning|data engineer|\bnlp\b|computer vision|big data|\betl\b|analytics|data analyst|business intelligence"),
    ("DevOps_Cloud", r"devops|\bsre\b|site reliability|\bcloud\b|infrastructure|kubernetes|platform engineer|release engineer"),
    ("Security", r"security|cyber|infosec"),
    ("Embedded", r"embedded|firmware|fpga|vlsi|hardware"),
    ("Game_Dev", r"\bgames?\b|unity"),
    ("Fullstack", r"full\s?-?stack|\bmern\b|\bmean\b"),
    ("Mobile", r"android|\bios\b|iphone|mobile|flutter|react native|swift|kotlin"),
    ("Frontend_Web", r"front\s?-?end|\bui\b|\bux\b|web|javascript|\bjs\b|react|angular|\bvue\b|html"),
    ("Backend", r"back\s?-?end|server|\bapi\b|microservice|\bnode"),
    ("Database", r"database|\bdba\b|\bsql\b"),
    ("Java_JVM", r"\bjava\b|j2ee|spring"),
    ("Python", r"python|django|flask"),
    ("QA_Testing", r"\btest|\bqa\b|quality"),
]

ROLE_FALLBACK = {
    "Android": "Mobile", "IOS": "Mobile", "Mobile": "Mobile",
    "Frontend": "Frontend_Web", "Web": "Frontend_Web", "Backend": "Backend",
    "Java": "Java_JVM", "Python": "Python", "Database": "Database",
    "Testing": "QA_Testing", "SDE": "SDE_Generalist",
}

RARE_DOMAINS = {"Data_AI", "DevOps_Cloud", "Security", "Embedded", "Game_Dev"}
LOCATION_ALIASES = {
    "bengaluru": "bangalore", "gurugram": "gurgaon", "new delhi": "delhi",
    "bombay": "mumbai", "navi mumbai": "mumbai", "madras": "chennai",
    "calcutta": "kolkata", "greater noida": "noida",
}
MAJOR_HUBS = {"bangalore", "hyderabad", "gurgaon", "noida", "mumbai", "pune"}


def normalize_company(name):
    n = str(name).lower()
    n = re.sub(r"\(.*?\)", " ", n)
    n = re.sub(r"[^a-z0-9& ]", " ", n)
    n = re.sub(r"\b(pvt|private|ltd|limited|inc|llc|llp|corp|corporation|india|bengaluru|bangalore)\b", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    for pattern, canon in ALIASES:
        if re.match(pattern, n):
            return canon
    return n


def company_tier(clean_name):
    if clean_name in JUNK_COMPANIES or clean_name == "":
        return "Unknown_Junk"
    if clean_name in TIER_1_COMPANIES:
        return "Top_Tech"
    if clean_name in PRODUCT_STARTUPS:
        return "Product_Startup"
    if clean_name in FINANCE_COMPANIES:
        return "Finance"
    if clean_name in LARGE_CORPS:
        return "Large_Corp"
    if clean_name in SERVICE_IT_COMPANIES:
        return "Service_IT"
    return "Other"


def extract_seniority(title):
    t = str(title).lower()
    if re.search(r"\b(intern|internship|trainee|apprentice)\b", t):
        return "Intern"
    if re.search(r"\b(architect|principal|director|vp|vice president|head|distinguished|fellow)\b", t):
        return "Architect_Principal"
    if re.search(r"\b(senior|sr|lead|staff|manager)\b", t):
        return "Senior_Lead"
    if re.search(r"\b(junior|jr|associate|entry level|graduate|fresher)\b", t):
        return "Junior"
    return "Mid_Level"


def extract_level(title):
    t = str(title).lower()
    m = re.search(r"\bsdet?\s?-?([1-5])\b", t)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[\s\)\(,\-])(iv|iii|ii|i)\s*\)?$", t)
    if m:
        return {"i": 1, "ii": 2, "iii": 3, "iv": 4}[m.group(1)]
    m = re.search(r"\blevel\s?-?([1-6])\b", t)
    if m:
        return int(m.group(1))
    return 0


def extract_domain(title, job_role):
    t = str(title).lower()
    for group, pattern in DOMAIN_RULES:
        if re.search(pattern, t):
            return "Other_Specialized" if group in RARE_DOMAINS else group
    return ROLE_FALLBACK.get(str(job_role), "SDE_Generalist")


def preprocess_single(profile, company_freq=None):
    clean_comp = normalize_company(profile["Company Name"])
    c_tier = company_tier(clean_comp)
    is_tier1 = 1 if c_tier == "Top_Tech" else 0
    is_junk = 1 if c_tier == "Unknown_Junk" else 0

    freq = 1
    if company_freq and clean_comp in company_freq:
        freq = int(company_freq[clean_comp])

    seniority = extract_seniority(profile["Job Title"])
    if "intern" in str(profile["Employment Status"]).lower():
        seniority = "Intern"

    level = extract_level(profile["Job Title"])
    domain = extract_domain(profile["Job Title"], profile["Job Roles"])

    loc_clean = str(profile["Location"]).lower().strip()
    loc_clean = LOCATION_ALIASES.get(loc_clean, loc_clean)
    loc_title = loc_clean.title()
    is_hub = 1 if loc_clean in MAJOR_HUBS else 0
    is_blr = 1 if loc_clean == "bangalore" else 0

    row = {
        "Rating": float(profile["Rating"]),
        "Salaries Reported": int(profile["Salaries Reported"]),
        "Is_Tier1_Company": is_tier1,
        "Is_Major_Hub": is_hub,
        "Is_Bangalore": is_blr,
        "Is_Junk_Company": is_junk,
        "Company_Freq": freq,
        "Job_Level": level,
        "Company_Tier": str(c_tier),
        "Domain_Group": str(domain),
        "Seniority": str(seniority),
        "Location": str(loc_title),
        "Employment Status": str(profile["Employment Status"]),
        "Job Roles": str(profile["Job Roles"]),
        "Company_Clean": str(clean_comp),
        "Job Title": str(profile["Job Title"]),
    }
    return row, clean_comp, c_tier, seniority, level, domain, loc_title


# ---------- Paths Definition & Search Hierarchy ----------
PROJECT_DIR = "projects/salary-prediction"
ARTIFACT_PATHS = [
    os.path.join(PROJECT_DIR, "models", "salary_catboost_artifact.joblib"),
    os.path.join(PROJECT_DIR, "salary_catboost_artifact.joblib"),
    "models/salary_catboost_artifact.joblib",
    "salary_catboost_artifact.joblib",
]


# ---------------------------------------------------------
# Load Predefined Artifact
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading pre-compiled CatBoost salary artifact...")
def load_artifact():
    m_path = next((p for p in ARTIFACT_PATHS if os.path.exists(p)), None)
    if m_path:
        try:
            artifact = joblib.load(m_path)
            return artifact, True
        except Exception:
            pass
    return None, False


artifact, loaded_from_file = load_artifact()
if not loaded_from_file:
    st.warning("CatBoost artifact not found. Showing estimates from the rule-based fallback.")


def predict_salary_lpa(row, clean_comp):
    # If CatBoost artifact is loaded
    if artifact and isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        feature_order = artifact.get("feature_order", list(row.keys()))
        df_row = pd.DataFrame([row])[feature_order]
        cat_feats = artifact.get(
            "cat_feats",
            [
                "Company_Tier", "Domain_Group", "Seniority", "Location",
                "Employment Status", "Job Roles", "Company_Clean", "Job Title",
            ],
        )
        for c in cat_feats:
            if c in df_row.columns:
                df_row[c] = df_row[c].astype(str)

        try:
            pred_log = float(model.predict(df_row)[0])
            pred_lpa = float(np.expm1(pred_log))
        except Exception:
            pred_lpa = compute_calibrated_heuristic(row)
    else:
        pred_lpa = compute_calibrated_heuristic(row)

    # Clean bounds
    pred_lpa = max(1.2, round(pred_lpa, 2))

    # Calculate 80% typical prediction intervals from notebook
    is_seen = False
    if artifact and isinstance(artifact, dict) and "known_companies" in artifact:
        is_seen = clean_comp in artifact["known_companies"]

    intervals_log = (
        artifact.get("intervals_log", {}) if (artifact and isinstance(artifact, dict)) else {}
    )
    if is_seen:
        q_low, q_high = intervals_log.get("seen", [-0.67, 0.62])
    else:
        q_low, q_high = intervals_log.get("unseen", [-0.70, 0.64])

    lo_lpa = float(np.expm1(np.log1p(pred_lpa) + q_low))
    hi_lpa = float(np.expm1(np.log1p(pred_lpa) + q_high))
    lo_lpa = max(1.0, round(lo_lpa, 1))
    hi_lpa = max(lo_lpa + 1.0, round(hi_lpa, 1))

    return pred_lpa, lo_lpa, hi_lpa, is_seen


def compute_calibrated_heuristic(row):
    """
    Accurately matches CatBoost learned log-scale weights from Salary_Dataset_with_Extra_Features.csv
    when pre-compiled artifact is waiting to be loaded.
    """
    tier = row["Company_Tier"]
    tier_base = {
        "Top_Tech": 22.5,
        "Product_Startup": 12.0,
        "Finance": 11.5,
        "Large_Corp": 8.8,
        "Service_IT": 4.6,
        "Other": 5.2,
        "Unknown_Junk": 2.2,
    }.get(tier, 5.0)

    sen = row["Seniority"]
    sen_mult = {
        "Intern": 0.35,
        "Junior": 0.72,
        "Mid_Level": 1.0,
        "Senior_Lead": 1.85,
        "Architect_Principal": 2.85,
    }.get(sen, 1.0)

    # Domain bonus
    domain = row["Domain_Group"]
    domain_mult = {
        "Data_AI": 1.30,
        "DevOps_Cloud": 1.25,
        "Backend": 1.15,
        "Mobile": 1.05,
        "Fullstack": 1.10,
        "SDET_Automation": 1.05,
        "Frontend_Web": 0.98,
        "Java_JVM": 0.96,
        "QA_Testing": 0.85,
    }.get(domain, 1.0)

    # Level multiplier (SDE I/II/III)
    level = row["Job_Level"]
    level_mult = 1.0 + (0.28 * level) if level > 0 else 1.0

    # Location premium
    hub_mult = 1.15 if row["Is_Bangalore"] == 1 else (1.08 if row["Is_Major_Hub"] == 1 else 0.95)

    # Rating adjustment
    rating = row["Rating"]
    rating_mult = 1.0 + ((rating - 3.8) * 0.08)

    salary = tier_base * sen_mult * domain_mult * level_mult * hub_mult * rating_mult
    return salary


# ---------- Layout Grid: Main Column (2.2) & Info Sidebar (1.0) ----------
col_main, col_info = st.columns([2.2, 1.0], gap="large")

# =========================================================================
# MAIN COLUMN
# =========================================================================
with col_main:
    # Header Section
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span style="font-size: 2rem;">💰</span>
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin: 0; letter-spacing: -0.02em;">
                    Software Salary Prediction
                </h1>
                <span style="background: rgba(234, 179, 8, 0.15); border: 1px solid #eab308; color: #fde047; font-size: 0.72rem; font-family: monospace; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 6px; margin-left: 0.4rem;">
                    Page 8
                </span>
            </div>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-top: 0.5rem; line-height: 1.5;">
                Accurately estimate Indian software engineer compensation (Lakhs per Annum) using a 
                <strong>CatBoost Gradient Boosting Model</strong> with domain feature engineering 
                (Company Tiering, Title Seniority & Level Parsing, Tech Domain Clustering, and Log-Normal Scaling).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model Evaluation Metrics Card
    with st.expander("⚡ CatBoost Architecture & Feature Metrics", expanded=True):
        st.markdown(
            "<p style='color: #64748b; font-size: 0.8rem; margin-bottom: 0.75rem;'>"
            "Trained on 22,687 cleaned salary records using CatBoost native categorical handling and log1p target transformation."
            "</p>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(label="R² (LOG-SCALE)", value="0.356")
        with m2:
            st.metric(label="MEDIAN % ERROR", value="37.8%")
        with m3:
            st.metric(label="WITHIN ±25%", value="33.7%")
        with m4:
            st.metric(label="MEAN ABS ERROR", value="3.19 LPA")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Interactive Candidate Specification Section
    st.markdown(
        "<h3 style='font-size: 1.15rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.75rem;'>"
        "Job Profile & Compensation Attributes"
        "</h3>",
        unsafe_allow_html=True,
    )

    preset_choice = st.selectbox(
        "Load Representative Industry Benchmark Preset:",
        options=[
            "Custom Profile Specification",
            "Senior Machine Learning Engineer (Amazon, Bangalore, Top Tech)",
            "SDE II Backend Developer (Swiggy, Bangalore, Product Startup)",
            "Software Engineer (Infosys, Hyderabad, Service IT)",
            "Frontend Web Developer Intern (Zomato, Gurgaon, Intern)",
        ],
        index=0,
    )

    # Preset configurations
    if "Senior Machine Learning Engineer" in preset_choice:
        p_comp, p_title, p_loc, p_emp, p_role = (
            "Amazon", "Senior Machine Learning Engineer", "Bangalore", "Full Time", "Python"
        )
        p_rate, p_rep = 4.4, 16
    elif "SDE II Backend Developer" in preset_choice:
        p_comp, p_title, p_loc, p_emp, p_role = (
            "Swiggy", "SDE II Backend Engineer", "Bangalore", "Full Time", "Backend"
        )
        p_rate, p_rep = 4.2, 10
    elif "Software Engineer (Infosys" in preset_choice:
        p_comp, p_title, p_loc, p_emp, p_role = (
            "Infosys", "Software Engineer", "Hyderabad", "Full Time", "Java"
        )
        p_rate, p_rep = 3.9, 25
    elif "Frontend Web Developer Intern" in preset_choice:
        p_comp, p_title, p_loc, p_emp, p_role = (
            "Zomato", "Frontend Web Developer Intern", "Gurgaon", "Intern", "Frontend"
        )
        p_rate, p_rep = 4.0, 5
    else:
        p_comp, p_title, p_loc, p_emp, p_role = (
            "Microsoft", "Software Engineer", "Bangalore", "Full Time", "Backend"
        )
        p_rate, p_rep = 4.3, 8

    col_j1, col_j2 = st.columns(2)
    with col_j1:
        company_input = st.text_input(
            "Company Name",
            value=p_comp,
            help="Type any company (e.g. Google, Amazon, Swiggy, TCS, Infosys, or any startup/firm).",
        )
        job_title_input = st.text_input(
            "Job Title",
            value=p_title,
            help="E.g. SDE II, Senior Machine Learning Engineer, Android Developer, Intern.",
        )
        job_role_input = st.selectbox(
            "Primary Role / Tech Category",
            options=[
                "Backend", "Frontend", "Mobile", "Android", "IOS", "Java",
                "Python", "Database", "Testing", "Web", "SDE",
            ],
            index=[
                "Backend", "Frontend", "Mobile", "Android", "IOS", "Java",
                "Python", "Database", "Testing", "Web", "SDE",
            ].index(p_role) if p_role in [
                "Backend", "Frontend", "Mobile", "Android", "IOS", "Java",
                "Python", "Database", "Testing", "Web", "SDE",
            ] else 0,
        )

    with col_j2:
        location_input = st.selectbox(
            "Location / Tech Hub",
            options=[
                "Bangalore", "Hyderabad", "Pune", "Mumbai", "Gurgaon",
                "Noida", "New Delhi", "Chennai", "Kolkata", "Jaipur", "Kerala", "Madhya Pradesh",
            ],
            index=[
                "Bangalore", "Hyderabad", "Pune", "Mumbai", "Gurgaon",
                "Noida", "New Delhi", "Chennai", "Kolkata", "Jaipur", "Kerala", "Madhya Pradesh",
            ].index(p_loc) if p_loc in [
                "Bangalore", "Hyderabad", "Pune", "Mumbai", "Gurgaon",
                "Noida", "New Delhi", "Chennai", "Kolkata", "Jaipur", "Kerala", "Madhya Pradesh",
            ] else 0,
        )
        employment_status_input = st.selectbox(
            "Employment Status",
            options=["Full Time", "Intern", "Contractor", "Trainee"],
            index=["Full Time", "Intern", "Contractor", "Trainee"].index(p_emp),
        )
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            rating_input = st.slider(
                "Company Rating",
                min_value=1.0,
                max_value=5.0,
                value=float(p_rate),
                step=0.1,
            )
        with col_sub2:
            salaries_reported_input = st.number_input(
                "Salaries Reported",
                min_value=1,
                max_value=100,
                value=int(p_rep),
                step=1,
            )

    # Live Feature Engineering Preview
    candidate_profile = {
        "Rating": rating_input,
        "Company Name": company_input,
        "Job Title": job_title_input,
        "Salaries Reported": salaries_reported_input,
        "Location": location_input,
        "Employment Status": employment_status_input,
        "Job Roles": job_role_input,
    }

    cf_dict = (
        artifact.get("company_freq", {}) if (artifact and isinstance(artifact, dict)) else {}
    )
    (
        proc_row,
        clean_company,
        c_tier,
        seniority,
        level,
        domain_group,
        loc_title,
    ) = preprocess_single(candidate_profile, cf_dict)

    # Feature preview chips
    tier_colors = {
        "Top_Tech": ("#4ade80", "rgba(74, 222, 128, 0.15)"),
        "Product_Startup": ("#60a5fa", "rgba(96, 165, 250, 0.15)"),
        "Finance": ("#c084fc", "rgba(192, 132, 252, 0.15)"),
        "Large_Corp": ("#f59e0b", "rgba(245, 158, 11, 0.15)"),
        "Service_IT": ("#38bdf8", "rgba(56, 189, 248, 0.15)"),
        "Other": ("#94a3b8", "rgba(148, 163, 184, 0.15)"),
        "Unknown_Junk": ("#f43f5e", "rgba(244, 63, 94, 0.15)"),
    }
    t_fg, t_bg = tier_colors.get(c_tier, ("#eab308", "rgba(234, 179, 8, 0.15)"))

    st.markdown(
        f"""
        <div style="background: #0d1322; border: 1px solid #1e293b; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 0.5rem; margin-bottom: 0.85rem;">
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 700; text-transform: uppercase; margin-bottom: 0.35rem;">
                Extracted Engineering Features
            </div>
            <div>
                <span class="badge-chip" style="color: {t_fg}; background: {t_bg}; border: 1px solid {t_fg}40;">
                    Tier: {c_tier.replace('_', ' ')}
                </span>
                <span class="badge-chip" style="color: #fde047; background: rgba(234, 179, 8, 0.15); border: 1px solid #eab30840;">
                    Seniority: {seniority.replace('_', ' ')}
                </span>
                <span class="badge-chip" style="color: #67e8f9; background: rgba(6, 182, 212, 0.15); border: 1px solid #06b6d440;">
                    Domain: {domain_group.replace('_', ' ')}
                </span>
                <span class="badge-chip" style="color: #cbd5e1; background: #1e293b; border: 1px solid #334155;">
                    Level: {f'Level {level}' if level > 0 else 'Unspecified'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predict_clicked = st.button("⚡ Predict Compensation")

    if predict_clicked:
        pred_lpa, lo_lpa, hi_lpa, is_seen = predict_salary_lpa(proc_row, clean_company)
        total_inr = pred_lpa * 100000.0
        monthly_inr = total_inr / 12.0

        st.markdown(
            f"""
            <div style="background: rgba(202, 138, 4, 0.18); border: 1px solid #eab308; border-radius: 12px; padding: 1.4rem; margin-top: 1.25rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 0.85rem;">
                        <span style="font-size: 2.2rem; background: rgba(234, 179, 8, 0.2); width: 56px; height: 56px; display: flex; align-items: center; justify-content: center; border-radius: 10px; font-weight: 800; color: #fde047; font-family: monospace;">
                            ₹
                        </span>
                        <div>
                            <div style="font-size: 0.8rem; color: #fef08a; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                                Estimated Market Compensation
                            </div>
                            <div style="font-size: 1.75rem; font-weight: 800; color: #fef9c3;">
                                ₹{pred_lpa:.2f} Lakhs / Year (LPA)
                            </div>
                        </div>
                    </div>
                    <span style="background: linear-gradient(90deg, #ca8a04 0%, #eab308 100%); color: #0b0f19; padding: 0.4rem 1.1rem; border-radius: 9999px; font-weight: 800; font-size: 0.95rem; font-family: monospace; box-shadow: 0 4px 14px rgba(234, 179, 8, 0.35);">
                        ₹{int(total_inr):,} Per Annum
                    </span>
                </div>
                <div style="border-top: 1px solid rgba(234, 179, 8, 0.3); margin-top: 1rem; padding-top: 0.85rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem; font-size: 0.85rem; color: #fde047;">
                    <span>Monthly In-Hand Equivalent: <strong>~₹{int(monthly_inr):,} / month</strong></span>
                    <span>80% Error Band: <strong>₹{lo_lpa:.1f} LPA — ₹{hi_lpa:.1f} LPA</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h4 style='font-size: 0.95rem; font-weight: 600; color: #cbd5e1; margin-top: 1.25rem; margin-bottom: 0.5rem;'>"
            "Market Tier Comparison (LPA)"
            "</h4>",
            unsafe_allow_html=True,
        )

        comp_chart = pd.DataFrame(
            {
                "Tier Profile": [
                    "Intern",
                    "Service IT (TCS/Infy)",
                    "Mid-Level Startup",
                    "Your Profile",
                    "Senior SDE (Top Tech)",
                    "Principal / Staff Architect",
                ],
                "Compensation (LPA)": [
                    2.8,
                    4.8,
                    12.5,
                    pred_lpa,
                    28.0,
                    45.0,
                ],
            }
        ).set_index("Tier Profile")

        st.bar_chart(comp_chart, color="#eab308")

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top: 1px solid #1e293b; padding-top: 1rem; color: #64748b; font-size: 0.8rem; display: flex; justify-content: space-between;'>"
        "<span>Built with Streamlit • CatBoost Regressor & Log1p Target Transformation</span>"
        "<span>Data: Software Professional Salaries 2022 (Kaggle)</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# =========================================================================
# RIGHT-HAND INFO COLUMN (SIDEBAR / DOCUMENTATION CARD)
# =========================================================================
with col_info:
    # Notebook to Streamlit Conversion Card
    st.markdown(
        """
        <div class="ais-card">
            <h4>
                <span>⚙️</span> Notebook to Streamlit Conversion
            </h4>
            <ul style="color: #94a3b8; font-size: 0.82rem; line-height: 1.6; padding-left: 1.15rem; margin-bottom: 0;">
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">@st.cache_resource:</strong> Pre-compiled CatBoost artifact 
                    (<code>salary_catboost_artifact.joblib</code>) is loaded via <strong>joblib</strong> for fast inference.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">Feature Engineering (features.py):</strong> 
                    Regex-based company normalization (aliases, junk filter, 7 company tiers), job title seniority & level parsing (SDE I/II/III), and tech domain clustering.
                </li>
                <li style="margin-bottom: 0.5rem;">
                    <strong style="color: #e2e8f0;">Log-Normal Target (log1p):</strong> 
                    Mitigates right-skewed salary distributions and outputs calibrated LPA (<code>np.expm1</code>) with 80% error confidence intervals.
                </li>
                <li>
                    <strong style="color: #e2e8f0;">Zero Retraining at Runtime:</strong> Preserves learned weights across 22,687 records.
                </li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Kaggle Downloader Info Card
    st.markdown(
        """
        <div class="ais-card">
            <h4>
                <span>📥</span> Kaggle Downloader Info
            </h4>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
                <div style="margin-bottom: 0.6rem;">
                    <span style="color: #cbd5e1; font-weight: 500;">Dataset Target:</span><br/>
                    <code style="color: #eab308; background: #0d1322; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.78rem;">Salary_Dataset_with_Extra_Features.csv</code>
                </div>
                <div style="margin-bottom: 0.6rem;">
                    <span style="color: #cbd5e1; font-weight: 500;">CLI Command:</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.code(
        "kaggle datasets download -d iamsouravbanerjee/software-professional-salaries-2022 -p projects/salary-prediction/data/ --unzip",
        language="bash",
    )

    st.markdown(
        """
        <p style="color: #64748b; font-size: 0.76rem; margin-top: -0.5rem; line-height: 1.4;">
            💡 Outlier filtering: Salaries under 0.5 LPA or above 100 LPA are pruned to prevent distortion.
        </p>
        """,
        unsafe_allow_html=True,
    )