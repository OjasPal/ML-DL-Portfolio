import re
import numpy as np
import pandas as pd

# ---------------- Companies ----------------
ALIASES = [
    (r"^(tcs|tata consultancy)", "tata consultancy services"),
    (r"^wipro", "wipro"), (r"^hcl", "hcl technologies"),
    (r"^cognizant", "cognizant"), (r"^infosys", "infosys"),
    (r"^accenture", "accenture"), (r"^capgemini", "capgemini"),
    (r"^(lti|ltimindtree|l t infotech|mindtree)\b", "ltimindtree"),
    (r"^ibm", "ibm"), (r"^dell", "dell technologies"),
    (r"^amazon", "amazon"), (r"^google", "google"), (r"^microsoft", "microsoft"),
    (r"^(facebook|meta platforms|meta)$", "meta"),
    (r"^cts\b", "cognizant"), (r"^samsung", "samsung"), (r"^walmart", "walmart"),
    (r"^(hewlett packard enterprise|hpe)", "hewlett packard enterprise"),
    (r"^(atos|syntel)", "atos"), (r"^dbs", "dbs bank"), (r"^(robert )?bosch", "bosch"),
    (r"^hashedin", "hashedin"), (r"^csc$", "dxc technology"),
    (r"^cisco", "cisco"), (r"^oracle", "oracle"), (r"^expedia", "expedia"),
    (r"^(j p morgan|jpmorgan|jp morgan)", "jpmorgan"),
    (r"^ntt data", "ntt data"), (r"^deloitte", "deloitte"),
    (r"^paytm", "paytm"), (r"^flipkart", "flipkart"),
    (r"^ola( electric| cabs)?$", "ola"),
    (r"^(byju s|byjus)", "byjus"), (r"^(cure fit|curefit)", "cure fit"),
]

JUNK_COMPANIES = {
    "fresher", "abc", "freelancer", "done by none", "anonymous", "xyz", "nones", "none",
    "employers", "test", "yes", "na", "n a", "self employed", "student", "first student",
    "fresherworld com", "anonymous content", "confidential", "private", "company", "startup",
    "abcdef", "freshers com", "freshers", "java", "pk", "xyz tech", "other", "others", "nil", "no company",
}

TIER_1_COMPANIES = {
    "google", "microsoft", "amazon", "meta", "apple", "uber", "cisco", "oracle", "flipkart",
    "swiggy", "zomato", "adobe", "linkedin", "salesforce", "atlassian",
    "netflix", "nvidia", "intel", "amd", "broadcom", "vmware", "paypal", "intuit", "qualcomm",
    "samsung", "sap", "servicenow", "workday", "expedia", "airbnb", "stripe", "databricks",
    "goldman sachs", "morgan stanley", "d e shaw", "de shaw", "tower research capital",
    "directi", "rubrik", "nutanix", "yahoo", "walmart",
}

PRODUCT_STARTUPS = {
    "paytm", "phonepe", "razorpay", "ola", "meesho", "dream11", "cred", "zerodha", "groww",
    "byjus", "unacademy", "inmobi", "sharechat", "mindtickle", "media net", "hike", "grab",
    "gojek", "hotstar", "hevo data", "joveo", "cure fit", "blinkit", "urban company", "urbanclap",
    "oyo", "rupeek", "gainsight", "myntra", "nykaa", "lenskart", "freshworks", "zoho",
    "postman", "browserstack", "chargebee", "delhivery", "bigbasket", "dunzo", "zepto",
    "upstox", "pine labs", "mobikwik", "juspay", "cars24", "spinny", "gameskraft",
    "housing com", "toppr com", "mysmartprice", "planful", "trilogy innovations", "mobileiron",
    "nortonlifelock", "groupon", "bookmyshow", "snapdeal", "freecharge", "times internet", "hashedin",
}

LARGE_CORPS = {
    "verizon", "honeywell", "siemens", "philips", "harman", "bosch", "akamai", "comcast",
    "optum", "hewlett packard enterprise", "bharti airtel", "jio", "ericsson", "nokia",
    "mcafee", "mckinsey", "pwc", "ey", "kpmg", "tesco",
}

FINANCE_COMPANIES = {
    "dbs bank",
    "jpmorgan", "bny mellon", "fis", "fiserv", "barclays", "deutsche bank", "citi", "citibank",
    "hsbc", "wells fargo", "american express", "visa", "mastercard", "state street",
    "standard chartered", "bank of america", "ubs", "credit suisse", "nomura",
    "fidelity investments", "blackrock",
}

SERVICE_IT_COMPANIES = {
    "tata consultancy services", "infosys", "wipro", "hcl technologies", "cognizant",
    "accenture", "capgemini", "tech mahindra", "ltimindtree", "dxc technology", "virtusa",
    "mphasis", "cgi", "ntt data", "itc infotech", "hexaware technologies", "zensar technologies",
    "birlasoft", "coforge", "niit technologies", "sonata software", "persistent systems",
    "l t technology services", "epam systems", "publicis sapient", "ibm", "dell technologies",
    "genpact", "wns", "globallogic", "raja software labs", "accolite digital", "infogain", "valuelabs",
    "photon", "diverse lynx", "teksystems", "collabera", "virtual employee", "atos", "firstsource", "syntel", "igate", "atos", "unisys", "ust global", "ust",
    "deloitte",
}


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


# ---------------- Titles ----------------
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
    """Numeric level: SDE I/II/III, SDE1/2/3, L4 ... (0 = not stated)."""
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


RARE_DOMAINS = {"Data_AI", "DevOps_Cloud", "Security", "Embedded", "Game_Dev"}  # too few rows in this dataset


def extract_domain(title, job_role):
    t = str(title).lower()
    for group, pattern in DOMAIN_RULES:
        if re.search(pattern, t):
            return "Other_Specialized" if group in RARE_DOMAINS else group
    return ROLE_FALLBACK.get(str(job_role), "SDE_Generalist")


LOCATION_ALIASES = {"bengaluru": "bangalore", "gurugram": "gurgaon", "new delhi": "delhi",
                    "bombay": "mumbai", "navi mumbai": "mumbai", "madras": "chennai",
                    "calcutta": "kolkata", "greater noida": "noida"}
MAJOR_HUBS = {"bangalore", "hyderabad", "gurgaon", "noida", "mumbai", "pune"}


def build_company_freq(df_clean):
    """Company -> row count, computed on TRAINING data. Save it with the model for the app."""
    return df_clean["Company_Clean"].value_counts().to_dict()


def preprocess_dataframe(df, company_freq=None):
    """company_freq=None -> computed from df (training). Pass the saved dict for single-row inference."""
    df = df.copy()
    df["Company_Clean"] = df["Company Name"].apply(normalize_company)
    df["Company_Tier"] = df["Company_Clean"].apply(company_tier)
    df["Is_Tier1_Company"] = (df["Company_Tier"] == "Top_Tech").astype(int)
    df["Is_Junk_Company"] = (df["Company_Tier"] == "Unknown_Junk").astype(int)
    if company_freq is None:
        df["Company_Freq"] = df.groupby("Company_Clean")["Company_Clean"].transform("count")
    else:
        df["Company_Freq"] = df["Company_Clean"].map(company_freq).fillna(1).astype(int)

    df["Seniority"] = df["Job Title"].apply(extract_seniority)
    status_intern = df["Employment Status"].astype(str).str.lower().str.contains("intern")
    df.loc[status_intern, "Seniority"] = "Intern"
    df["Job_Level"] = df["Job Title"].apply(extract_level)
    df["Domain_Group"] = [extract_domain(t, r) for t, r in zip(df["Job Title"], df["Job Roles"])]

    loc = df["Location"].astype(str).str.lower().str.strip().replace(LOCATION_ALIASES)
    df["Location"] = loc.str.title()
    df["Is_Major_Hub"] = loc.isin(MAJOR_HUBS).astype(int)
    df["Is_Bangalore"] = (loc == "bangalore").astype(int)
    return df