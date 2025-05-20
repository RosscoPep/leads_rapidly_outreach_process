import pandas as pd
import json
import time
import requests
import backoff
import logging
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any, Union
import concurrent.futures
from dotenv import load_dotenv
from enum import Enum, auto
from tqdm import tqdm
import re
import io

# --- Use Case Definitions (kept for context) ---
ALL_USE_CASES = [
    {
        "id": "outreach",
        "name": "Deep Personalization Cold Outreach System",
        "description": "Automates highly personalized outreach using AI analysis of public data, boosting reply rates and saving sales time.",
        "focus": ["Sales", "Marketing", "Growth"],
        "benefit_hint": "boost reply rates (e.g., 20-40% improvement), increase booked meetings",
        "keywords": ["sales", "outreach", "lead generation", "crm", "personalization", "growth"]
    },
    {
        "id": "knowledge",
        "name": "Automated Internal Knowledge Base Q&A",
        "description": "Deploys an AI chatbot trained on internal docs allowing employees to get instant answers, saving time and speeding up onboarding.",
        "focus": ["HR", "Operations", "IT", "General"],
        "benefit_hint": "save employee time searching for info, speed up onboarding",
        "keywords": ["knowledge base", "chatbot", "faq", "internal documentation", "employee support"]
    }
    # ... (rest of ALL_USE_CASES - kept for brevity in this display)
]
USE_CASE_DETAILS = {uc["id"]: uc for uc in ALL_USE_CASES}
PERSONA_TO_RELEVANT_USE_CASES = {
    "The Scaler": ["outreach", "pipeline", "qualification", "onboarding", "reporting"],
    "The Cost Cutter/Optimizer": ["invoice", "qualification", "reporting", "support", "transcription"],
    "The Innovator/Builder": ["content", "knowledge", "outreach", "proposal"],
    "The Risk Mitigator": ["support", "invoice", "social", "onboarding"],
    "Innovator/Trendsetter": ["content", "outreach", "social", "knowledge"],
    "Pragmatist/Problem-Solver": ["qualification", "invoice", "support", "reporting", "transcription"],
    "Business Outcomes Driver": ["outreach", "pipeline", "invoice", "reporting", "qualification"],
    "People/Culture Advocate": ["onboarding", "knowledge", "recruiting", "transcription"],
    "Technical Implementer": ["knowledge", "reporting", "qualification", "support"],
    "Strategic Leader": ["reporting", "outreach", "pipeline", "social"],
    "Operational Manager": ["invoice", "qualification", "reporting", "support", "onboarding", "transcription"],
    "Growth Driver": ["outreach", "pipeline", "qualification", "content", "proposal"],
    "General Professional": ["content", "transcription", "reporting", "knowledge"]
}

# Define error types
class ErrorType(Enum):
    PAYLOAD_CREATION = auto()
    API_CALL = auto()
    FUTURE_EXCEPTION = auto()
    JSON_PARSING = auto()
    MISSING_PERSONA = auto()
    SHORT_NAME_ERROR = auto()
    MISSING_DATA_POINTS = auto()
    LINKEDIN_DATA_ERROR = auto()

# --- Dynamic Path Configuration ---
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = Path(sys.executable).parent
else:
    SCRIPT_DIR = Path(__file__).parent

CONFIG_PATH = SCRIPT_DIR / 'config.yaml'
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ERROR: Configuration file not found at {CONFIG_PATH}. Please ensure 'config.yaml' exists.")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"ERROR: Could not parse configuration file 'config.yaml': {e}")
    sys.exit(1)


INPUT_FILE_NAME = config['files']['input']['name']
INPUT_SHEET_NAME = config['files']['input']['sheet']
OUTPUT_FILE_NAME = config['files']['output']['name']
LOG_FILE = config['files']['log']['name']
DOTENV_PATH = SCRIPT_DIR / '.env'
OUTPUT_DIR = SCRIPT_DIR / config['files']['output']['directory']
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE_PATH = SCRIPT_DIR / INPUT_FILE_NAME
OUTPUT_FILE_PATH = OUTPUT_DIR / OUTPUT_FILE_NAME

# LinkedIn Posts File Configuration
LI_POSTS_CONFIG = config['files'].get('linkedin_posts', {})
LI_POSTS_DIR = SCRIPT_DIR / LI_POSTS_CONFIG.get('directory', 'input')
LI_POSTS_FILE_NAME = LI_POSTS_CONFIG.get('name')
LINKEDIN_POSTS_FILE_PATH = LI_POSTS_DIR / LI_POSTS_FILE_NAME if LI_POSTS_FILE_NAME else None

# Fix for Unicode encoding in Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO, # Set to logging.DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        logging.FileHandler(SCRIPT_DIR / LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout) # sys.stdout is now UTF-8 wrapped
    ]
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=DOTENV_PATH)

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', config['gemini']['model_name'])
GEMINI_API_URL = config['gemini']['api_url'].format(model_name=GEMINI_MODEL_NAME)
REQUESTS_PER_MINUTE = int(os.getenv('GEMINI_RPM', config['gemini']['requests_per_minute']))
SECONDS_PER_REQUEST = 60.0 / REQUESTS_PER_MINUTE
MAX_WORKERS = int(os.getenv('MAX_WORKERS', config['gemini']['max_workers']))
SAFETY_SETTINGS = config['gemini']['safety_settings']
GENERATION_CONFIG = config['gemini']['generation_config']
PERSONA_GUIDANCE = config['persona_guidance']

# --- Runtime Configuration ---
TEST_MODE = config.get('runtime', {}).get('test_mode', True)
MAX_TEST_ROWS = config.get('runtime', {}).get('max_test_rows', 5)
CONSECUTIVE_ERROR_THRESHOLD = config.get('runtime', {}).get('consecutive_error_threshold', 10)
ERROR_PERCENTAGE_THRESHOLD = config.get('runtime', {}).get('error_percentage_threshold', 20)
DRY_RUN = config.get('runtime', {}).get('dry_run', False)
FUTURE_TIMEOUT = config.get('runtime', {}).get('future_timeout', 180)

# --- Configuration for Shortened Company Name ---
REMOVABLE_TRAILING_WORDS = [ # This list seems unused, direct logic is in get_shortened_company_name
    "Pty Ltd.", "Pty Ltd", "Ltd.", "Ltd", "Limited", "Inc.", "Inc", "Incorporated",
    "LLC.", "LLC", "Llc.", "Llc", "PLC.", "PLC", "Plc.", "Plc", "Corp.", "Corp", "Corporation",
    "GmbH", "AG", "SAS", "Co.", "Co", "Company", "Group", "Holdings", "Ventures", "Partners"
]
MAX_WORDS_FOR_SHORT_NAME = 2


# --- Helper Functions ---
def check_dependencies() -> None:
    missing_deps = []
    try:
        import openpyxl
    except ImportError:
        missing_deps.append('openpyxl')
    if missing_deps:
        msg = f"Missing dependencies: {', '.join(missing_deps)}. Please install using: pip install {' '.join(missing_deps)}"
        logger.error(msg)
        raise ImportError(msg)

def safe_str(value: Any) -> str:
    return str(value) if pd.notna(value) else ""

def parse_data_points(json_string: Optional[str]) -> List[Dict[str, Any]]:
    if not json_string or pd.isna(json_string):
        return []
    try:
        if isinstance(json_string, str):
            cleaned_json = json_string.strip().strip('\ufeff').strip()
            if cleaned_json in ('[]', '', '{}'): return []
            if not (cleaned_json.startswith('[') or cleaned_json.startswith('{')):
                logger.debug(f"JSON string doesn't appear to be valid JSON array or object: {cleaned_json[:50]}...")
                return []
            try:
                data = json.loads(cleaned_json)
            except json.JSONDecodeError:
                if cleaned_json.startswith('{') and cleaned_json.endswith('}'):
                    try: data = json.loads(f"[{cleaned_json}]")
                    except json.JSONDecodeError: return []
                else: return []
        elif isinstance(json_string, (list, dict)): # Already parsed, e.g. from JSON file
            data = json_string
        else: # Try to convert to string then parse
             data = json.loads(str(json_string))

        if isinstance(data, list): return data
        elif isinstance(data, dict): return [data]
        else: return []
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        log_str = str(json_string); log_str = (log_str[:97] + "...") if len(log_str) > 100 else log_str
        logger.warning(f"Could not parse data points JSON: {log_str} Error: {e}")
        return []

def get_persona_guidance(persona: str, row_index: int) -> Tuple[str, bool]:
    guidance = PERSONA_GUIDANCE.get(persona)
    if not guidance:
        logger.warning(f"Persona '{persona}' not found in PERSONA_GUIDANCE. Using 'General Professional'. Row index: {row_index}")
        return PERSONA_GUIDANCE.get("General Professional", "General guidance for professionals."), True # Added default for General Professional
    return guidance, False

def create_error_message(error_type: ErrorType, details: str = "") -> str:
    return f"ERROR_{error_type.name}: {details}"

def get_shortened_company_name(full_name: str) -> str:
    if not full_name or pd.isna(full_name): return ""
    name_to_process = str(full_name).strip()
    if not name_to_process: return ""

    # 1. Insert space before legal suffix if missing (e.g. ABCLimited -> ABC Limited)
    legal_suffixes = [
        "Pty Ltd", "Ltd", "Limited", "Inc", "Incorporated", "LLC", "PLC", "Corp", "Corporation",
        "GmbH", "AG", "SAS", "Co", "Company", "Group", "Holdings", "Ventures", "Partners"
    ]
    for suffix in legal_suffixes:
        # Add optional dot and case-insensitive
        name_to_process = re.sub(rf'(?i)([A-Za-z])({suffix})([\s\.]|$)', r'\1 \2\3', name_to_process)

    # 2. Replace '&' with 'and'
    name_to_process = name_to_process.replace('&', 'and')

    # 3. Convert all-caps names to title case unless short (likely acronym)
    if name_to_process.isupper() and len(name_to_process) > 4:
        name_to_process = name_to_process.title()
    # Also handle cases like 'ADDICTION RECOVERYLTD.'
    if name_to_process == name_to_process.upper():
        name_to_process = name_to_process.title()

    name_lower = name_to_process.lower()

    # Specific overrides
    if "trinity community arts" in name_lower: return "Trinity"
    if "bio-leadership project" in name_lower: return "The Bio-Leadership Project"

    # Heuristic for "X Communications" where X is multi-word
    if "communications" in name_lower and len(name_to_process.split()) > 2:
        parts = name_to_process.split()
        comm_index = next((i for i, part in enumerate(parts) if "communication" in part.lower()), -1)
        if comm_index > 1: return " ".join(parts[:comm_index])

    # 4. Remove common legal suffixes and some generic business types using regex for robustness
    legal_suffixes_pattern = r'\s+(Pty Ltd\.?|Ltd\.?|Limited|Inc\.?|Incorporated|LLC\.?|PLC\.?|Corp\.?|Corporation|GmbH|AG|SAS|Co\.?|Company|Group|Holdings|Ventures|Partners|Technologies|Technology|Tech|Systems|Solutions|Services|Software|International|Global|Worldwide|Consulting|Advisory|Consultants|Studio|Studios|Labs|Agency|Industries|Logistics|Enterprises|Manufacturing|Digital|Media|Online|Network|Analytics|Data|Research|Trading|Capital|Investments|Properties|Finance|Foundation|Institute|Institution|Organization|Council|Union|Association|Centre|Center|Fund|Trust|University|College|School|Academy|Associates|Bureau|Office|Department|Division|CIC|CIO|LLP\.?|Airport|Bank)(\s+|$|\.)'
    cleaned_name = re.sub(legal_suffixes_pattern, '', name_to_process, flags=re.IGNORECASE).strip()

    # If after stripping, the name is empty or just a suffix, revert to a simpler rule
    if not cleaned_name or cleaned_name.lower() in [suffix.lower().replace('.', '').replace(' ', '') for suffix_list in REMOVABLE_TRAILING_WORDS for suffix in suffix_list.split()]:
        name_parts_original = name_to_process.split()
        return " ".join(name_parts_original[:MAX_WORDS_FOR_SHORT_NAME]) if len(name_parts_original) > 0 else ""

    name_parts = cleaned_name.split()
    if not name_parts:
        return " ".join(name_to_process.split()[:MAX_WORDS_FOR_SHORT_NAME])

    # If three words, keep all unless the last is a legal suffix or generic type
    connectors = {"and", "of", "for", "the", "at", "by", "in", "on"}
    if len(name_parts) == 3:
        # If the last word is a legal suffix or generic type, drop it
        legal_suffixes_set = set([s.lower().replace('.', '') for s in legal_suffixes])
        if name_parts[2].lower().replace('.', '') in legal_suffixes_set:
            return " ".join(name_parts[:2])
        # Otherwise, keep all three
        return " ".join(name_parts)

    # If more than three words, shorten to two unless the second word is a connector
    if len(name_parts) > 3:
        if name_parts[1].lower() in connectors:
            return " ".join(name_parts[:3])
        return " ".join(name_parts[:2])

    # If two or fewer words, return as is
    return " ".join(name_parts)

def format_linkedin_post_age(post_timestamp: Union[datetime, pd.Timestamp]) -> str:
    if pd.isna(post_timestamp):
        return "recently"
    # Ensure post_timestamp is a datetime object
    if isinstance(post_timestamp, pd.Timestamp):
        post_timestamp = post_timestamp.to_pydatetime()

    now = datetime.now(post_timestamp.tzinfo) # Use post's timezone for "now" if available
    delta = now - post_timestamp

    if delta.days < 0: # Post is in the future? Log and return 'recently'
        logger.warning(f"Post timestamp {post_timestamp} is in the future compared to now {now}.")
        return "recently"
    if delta.days == 0:
        return "today"
    elif delta.days == 1:
        return "yesterday"
    elif delta.days < 7:
        return f"{delta.days} days ago"
    elif delta.days < 30:
        weeks = delta.days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    elif delta.days < 365:
        months = delta.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    else:
        years = delta.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"

def sanitized_profile_url(url: Any) -> str:
    if pd.isna(url):
        return ""
    url_str = str(url).strip()
    url_str = re.sub(r'^@', '', url_str)  # Remove leading @
    url_str = url_str.replace("https://www.", "").replace("http://www.", "").replace("https://", "").replace("http://", "")
    url_str = url_str.rstrip('/')
    url_str = url_str.split('?', 1)[0].split('#', 1)[0]
    # Specific common LinkedIn URL patterns
    # e.g., linkedin.com/in/username/ -> username
    # e.g., linkedin.com/pub/firstname-lastname/id -> pub/firstname-lastname/id (less common now)
    match = re.search(r'linkedin\.com/(?:in/|pub/|company/)?([^/]+)', url_str, re.IGNORECASE)
    if match and match.group(1):
        # Further clean the extracted part if it's a common profile ID
        # This part might need refinement based on typical garbage in profile URLs
        profile_id = match.group(1)
        # Remove trailing elements that are sometimes part of public URLs but not the core ID
        profile_id = profile_id.split('/')[0] # Takes the first part if there are more slashes
        return profile_id.lower()
    return url_str.lower() # Fallback to the cleaned full string if no specific LI pattern matches

# --- Modified Request Payload Creation ---
def create_gemini_request_payload(row: pd.Series) -> Optional[Dict[str, Any]]:
    try:
        person_name = safe_str(row.get('first_name'))
        company_name = safe_str(row.get('companyName'))
        job_title = safe_str(row.get('verifiedJobTitle'))
        assigned_persona = safe_str(row.get('assignedPersona', 'General Professional'))
        individual_summary = safe_str(row.get('individualActivitySummary', 'N/A'))
        company_summary = safe_str(row.get('companyNewsSummary', 'N/A'))
        shortened_company_name = safe_str(row.get('shortenedCompanyName', company_name))

        linkedin_posts_str = safe_str(row.get('formatted_linkedin_posts', "No specific recent LinkedIn posts provided for this contact.\n"))
        if not linkedin_posts_str.strip() or "No specific recent LinkedIn posts" in linkedin_posts_str :
             linkedin_posts_section = "**Recent LinkedIn Activity (Highest Priority for Observation/Icebreaker):**\nNo specific recent posts provided for this contact.\n"
        else:
             linkedin_posts_section = f"**Recent LinkedIn Activity (Highest Priority for Observation/Icebreaker):**\n{linkedin_posts_str}\n"

        if pd.isna(row.get('referenceableDataPoints')) and "No specific recent LinkedIn posts" in linkedin_posts_section:
            logger.warning(f"Missing referenceableDataPoints AND no LinkedIn posts for Name: {person_name}, Company: {company_name}. Payload quality will be low.")
            # Still proceed as per original logic, but quality depends on summaries

        data_points = parse_data_points(row.get('referenceableDataPoints'))

        guidance, used_fallback = get_persona_guidance(assigned_persona, row.name)
        if used_fallback:
            if not hasattr(create_gemini_request_payload, 'fallback_count'):
                create_gemini_request_payload.fallback_count = 0
            create_gemini_request_payload.fallback_count += 1

        relevant_points_web_search_str = ""
        if data_points:
            points_to_use = data_points[:3] # Max 3 web search points
            for i, point in enumerate(points_to_use):
                relevant_points_web_search_str += f"{i+1}. Type: {point.get('dataType', 'N/A')}, Detail: {point.get('detail', 'N/A')} (Source: {point.get('sourceHint', 'N/A')}, Recency: {point.get('recencyIndicator', 'N/A')})\n"
        
        if not relevant_points_web_search_str: # Ensure section header even if empty
            relevant_points_web_search_str = "None specific found from web search.\n"
        
        relevant_points_web_search_section = f"**Key Reference Points Found (from web search):**\n{relevant_points_web_search_str}"


        relevant_use_case_ids = PERSONA_TO_RELEVANT_USE_CASES.get(assigned_persona, PERSONA_TO_RELEVANT_USE_CASES["General Professional"])
        use_case_prompt_section = "Context about AI Accelerator's Offerings (Potential AI Use Cases we help with):\n"
        for i, uc_id in enumerate(relevant_use_case_ids[:3], 1): # Max 3 use cases
            uc_detail = USE_CASE_DETAILS.get(uc_id)
            if uc_detail:
                use_case_prompt_section += f"{i}. **{uc_detail['name']}:** {uc_detail['description']} (Benefit hint: {uc_detail['benefit_hint']})\n"

        # Construct the prompt (same as provided, ensure all variables are correctly populated)
        final_user_prompt = f"""**Role:** You are an expert cold email writer and applied psychologist specializing in crafting highly personalized, concise, and engaging components for B2B cold outreach emails to maximize positive engagement. Your business is 'AI Accelerator', an AI training and consultancy company based in England.

**Goal:** Generate FOUR components for an outreach email to {person_name} ({job_title}) at {company_name} (which will be referred to as {shortened_company_name} in parts of the email):
1.  `personalisedIcebreaker`: A compelling, natural opening line (1-2 sentences).
2.  `specificObservation`: A lead-in phrase starting with "Based on..." referring to an observation about their company or work.
3.  `specificPainPoint`: A statement that continues directly from the observation, identifying a potential pain point or priority for {shortened_company_name}.
4.  `impliedBelief`: A statement that begins with "I thought it may be useful given..." followed by a belief or value they likely hold.

**Context Provided:**
*   **Prospect Details:** {person_name}, who is {job_title} at {company_name} (referred to as {shortened_company_name}).
*   **Assigned Persona for Prospect:** {assigned_persona}
*   **Guidance for this Persona:** {guidance}
*   {linkedin_posts_section}
*   {relevant_points_web_search_section}
*   **Other Contextual Information:**
    *   Individual Activity Summary (LinkedIn Profile): {individual_summary}
    *   Company News Summary (Company Profile/Web): {company_summary}
*   {use_case_prompt_section}

**Detailed Instructions for Each Component:**

1.  **Data Prioritization and Usage Strategy (VERY IMPORTANT):**
    *   **For `specificObservation` and `personalisedIcebreaker`:**
        *   **Your primary source should be 'Recent LinkedIn Activity'.** Select the most recent, specific, and engaging post content.
        *   If no suitable LinkedIn posts are available or they are not relevant, use 'Key Reference Points Found (from web search)', prioritizing recency and specificity.
        *   As a third option, if the above are weak, you may draw a key theme or statement from the 'Individual Activity Summary' or 'Company News Summary'.
        *   Always aim for the most current and specific information available. The `personalisedIcebreaker` should feel like a natural continuation or reaction to the chosen observation.
    *   **For `specificPainPoint` and `impliedBelief`:**
        *   These must logically follow from the chosen `specificObservation`.
        *   Synthesize the observation with the prospect's `job_title`, `assignedPersona`, broader company context (using {shortened_company_name}), and the nature of AI Accelerator's offerings to infer a plausible pain point and a relevant underlying belief.
    *   **Cohesion is Key:** All four generated components must feel connected and build upon each other.

2.  **For `personalisedIcebreaker` (1-2 sentences):**
    *   Make it natural, engaging and directly relevant to the prospect's profile or recent activity (based on the prioritization above).
    *   It should sound natural, human, and genuinely engaging. Avoid generic compliments or overly salesy language.
    *   **Crucially, do NOT mention your company name ("AI Accelerator") or your product/service in the icebreaker itself.**
    *   Adopt the TONE and FOCUS described in the 'Persona Guidance'.
    *   NEVER include {person_name}'s first name in the icebreaker.
    *   Examples: "The RFU England kit launch with Umbro looked fantastic – a really strong campaign."; "I saw your recent LinkedIn post about managing difficult clients – some really insightful points there."; "Read the news about {shortened_company_name}'s recent funding round - congratulations!"; "I was just reading about Consol's approach to market leadership."

3.  **For `specificObservation`:**
    *   Start with "Based on..." followed by a reference to something specific you observed (from LinkedIn post, web search, or summary as per prioritization).
    *   This is a lead-in to the pain point.
    *   Examples: "Based on your recent post about X,"; "Based on {shortened_company_name}'s recent launch event,"; "Based on the news about {shortened_company_name} opening a new branch in San Fran,".

4.  **For `specificPainPoint`:**
    *   This should complete the thought started in `specificObservation`, forming a natural sentence.
    *   Must include the format: "I imagine that [pain point] is a priority."
    *   The pain point should be something that AI Accelerator's framework could implicitly help address.
    *   NEVER include {shortened_company_name} in the pain point.
    *   Examples: "I imagine that ensuring maximum ROI from those high-profile campaigns is a key priority."; "I imagine that one of your key priorities is optimizing internal processes to maintain a competitive edge."; "I imagine that mentoring a new team while ensuring delivery will be a key focus for you."; "I imagine that it's going to be tough managing your time while setting up the new office."

5.  **For `impliedBelief`:**
    *   Must start with "I thought it may be useful given..." followed by a belief or value they likely hold.
    *   This belief should subtly align with the value proposition of exploring innovative solutions like strategic AI.
    *   NEVER include {shortened_company_name} in the implied belief.
    *   Examples: "I thought it may be useful because you may need to do more with less resources right now." or "I thought it may be useful given you probably value strategies that optimise campaign performance." or "I thought it may be useful given your focus on strategic approaches to growth."; "I thought it may be may be useful given you probably need to think about how your business model may need to adapt with these new technologies."

6.  **Overall Tone & Language:**
    *   Maintain a respectful, professional, yet human and slightly informal tone.
    *   Ensure the components flow logically.
    *   **To enhance authenticity and avoid sounding overly robotic or like generic marketing copy, please refrain from using common AI clichés or overused buzzwords. For example, try to avoid words such as: 'delve', 'clearly', 'leverage', 'harness', 'seamless(ly)', 'robust', 'game-changer', 'unlock potential', 'paradigm shift', 'synergy', 'deep dive'. Instead, opt for more direct, natural, and specific language that sounds human.**
    *   **Fallback:** If, after reviewing all context, no strong, specific, and recent data point emerges for a truly personalized `specificObservation` from LinkedIn or web search, you may make a more general observation based on their `job_title`, `assignedPersona`, or `Individual Activity Summary` within their industry. However, always strive for specificity from the prioritized data sources first.

**Output Format:** Respond in JSON format ONLY, with NO additional text, greetings, or explanations outside the JSON structure. The JSON should look exactly like these examples, using the provided `{shortened_company_name}` where appropriate:

Example 1 (using {shortened_company_name} 'MLG'):
{{"personalisedIcebreaker": "The RFU England kit launch with Umbro looked fantastic – a really strong campaign.", "specificObservation": "Based on the recent launch,", "specificPainPoint": "I imagine that ensuring maximum ROI from those high-profile campaigns is a key priority for MLG.", "impliedBelief": "I thought it may be useful given you probably value strategies that optimise campaign performance."}}

Example 2 (using {shortened_company_name} 'Consol'):
{{"personalisedIcebreaker": "I saw your recentpost about the Real Estate conference where you were presenting at the Palma Marina - looked great!", "specificObservation": "Based on Consol's recent news, it looks like you've been focused on growing internally.", "specificPainPoint": "I imagine that a key priority for Consol is optimizing internal processes to maintain your competitive edge without significant external distractions.", "impliedBelief": "I thought it may be useful given you probably value strategic approaches."}}
"""
        contents = [
            {"role": "user", "parts": [{"text": "You are a helpful assistant who is an expert cold email writer. Your business is 'AI Accelerator'. Respond in JSON format only, like this: {\"personalisedIcebreaker\": \"...\", \"specificObservation\": \"...\", \"specificPainPoint\": \"...\", \"impliedBelief\": \"...\"}"}]},
            {"role": "model", "parts": [{"text": "OK. Provide the context (prospect details, persona, data points, LinkedIn activity etc.) and I will generate the personalisedIcebreaker, specificObservation, specificPainPoint, and impliedBelief in the specified JSON format, using the provided shortened company name where indicated."}]},
            {"role": "user", "parts": [{"text": final_user_prompt}]}
        ]
        payload = {
            "contents": contents,
            "safetySettings": SAFETY_SETTINGS,
            "generationConfig": {**GENERATION_CONFIG, "maxOutputTokens": 450}
        }
        return payload
    except Exception as e:
        row_identifier = f"Name: {row.get('first_name', 'N/A')}, Company: {row.get('companyName', 'N/A')}, Index: {row.name}"
        logger.error(f"Error creating Gemini request payload for {row_identifier}: {str(e)}", exc_info=True)
        return None

def process_row_for_gemini(index_row_tuple: Tuple[int, pd.Series], api_key: str) -> Tuple[int, str, str, str, str]:
    index, row = index_row_tuple
    error_msg_generic = create_error_message(ErrorType.PAYLOAD_CREATION, "Failed to create payload")
    error_msg_missing_data = create_error_message(ErrorType.MISSING_DATA_POINTS, "Missing referenceableDataPoints and valid LinkedIn Posts")

    if DRY_RUN:
        logger.info(f"DRY RUN: Using mock data for row index {index}")
        return (index, *get_mock_api_response(index))

    try:
        # Basic data check before creating payload (though payload creation also handles this)
        has_web_data = row.get('referenceableDataPoints') and not pd.isna(row.get('referenceableDataPoints')) and str(row.get('referenceableDataPoints')).strip() not in ('[]', '{}', 'None', 'nan', 'N/A', 'na')
        has_li_data = row.get('formatted_linkedin_posts') and "No specific recent LinkedIn posts" not in str(row.get('formatted_linkedin_posts'))

        if not has_web_data and not has_li_data:
             logger.warning(f"Skipping API call for row index {index} (Name: {row.get('first_name', 'N/A')}) due to missing both web data points and usable LinkedIn posts.")
             return index, error_msg_missing_data, error_msg_missing_data, error_msg_missing_data, error_msg_missing_data

        payload = create_gemini_request_payload(row)
        if payload:
            api_results = call_gemini_api(payload, api_key, row_index=index) # Pass index for logging
            time.sleep(SECONDS_PER_REQUEST) # Rate limiting
            return index, api_results[0], api_results[1], api_results[2], api_results[3]
        else:
            logger.warning(f"Skipping row index {index} due to payload creation error (payload was None).")
            # Distinguish if it's due to missing data points specifically (already checked above, but as a fallback)
            if not has_web_data and not has_li_data:
                return index, error_msg_missing_data, error_msg_missing_data, error_msg_missing_data, error_msg_missing_data
            else:
                return index, error_msg_generic, error_msg_generic, error_msg_generic, error_msg_generic
    except Exception as e:
        logger.error(f"Error processing row index {index}: {e}", exc_info=True)
        error_msg = create_error_message(ErrorType.FUTURE_EXCEPTION, str(e))
        return index, error_msg, error_msg, error_msg, error_msg

def should_retry_gemini(exception: Exception) -> bool:
    if isinstance(exception, requests.exceptions.RequestException):
        if hasattr(exception.response, 'status_code') and exception.response is not None:
            status_code = exception.response.status_code
            if status_code == 429: # Rate limit
                 retry_after = exception.response.headers.get("Retry-After")
                 wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 30 # Default 30s
                 logger.warning(f"Rate limit hit (429). Retrying after {wait_time} seconds...")
                 time.sleep(wait_time) # Backoff library handles expo, this is for immediate 429
                 return True
            if status_code in [500, 502, 503, 504]: # Server errors
                logger.warning(f"Server error {status_code}. Retrying...")
                return True
            if 400 <= status_code < 500 and status_code != 429: # Client errors (not 429)
                logger.error(f"Client error {status_code}. Not retrying. Response: {getattr(exception.response, 'text', 'No Response Body')[:500]}")
                return False
        return True # Default to retry for other request exceptions (e.g., connection errors)
    # For non-requests exceptions, e.g., JSON parsing errors from Gemini (if they were to be retried)
    # For now, we don't retry these, but this is where such logic would go.
    return False # Do not retry unknown exceptions by default

@backoff.on_exception(
    backoff.expo,
    Exception, # Catching general Exception to be handled by should_retry_gemini
    max_tries=5,
    giveup=lambda e: not should_retry_gemini(e),
    on_backoff=lambda details: logger.warning(f"Retrying API call for row index {details.get('args')[2] if len(details.get('args',[])) > 2 else 'N/A'}. Attempt {details['tries']} after {details['wait']:.1f}s due to {type(details['exception']).__name__}"),
    on_giveup=lambda details: logger.error(f"API call failed for row index {details.get('args')[2] if len(details.get('args',[])) > 2 else 'N/A'} after {details['tries']} attempts due to {type(details['exception']).__name__}: {details['exception']}")
)
def call_gemini_api(payload: Dict, api_key: str, row_index: Optional[int] = None) -> Tuple[str, str, str, str]: # Added row_index for logging
    error_tuple_key = (create_error_message(ErrorType.API_CALL, "API Key missing"),) * 4
    if not api_key:
        logger.error(f"Row {row_index}: Google API Key is missing.")
        return error_tuple_key
    error_tuple_payload = (create_error_message(ErrorType.API_CALL, "Empty payload"),) * 4
    if not payload:
        logger.error(f"Row {row_index}: Cannot call API with empty payload.")
        return error_tuple_payload

    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    try:
        logger.debug(f"Row {row_index}: Making API call to {GEMINI_API_URL}")
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload, timeout=FUTURE_TIMEOUT - 5) # Slightly less than future timeout
        logger.debug(f"Row {row_index}: API Response Status: {response.status_code}")
        response.raise_for_status()

        try:
            response_json = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Row {row_index}: Failed to parse API response as JSON. Error: {e}. Response text: {response.text[:500]}")
            return (create_error_message(ErrorType.JSON_PARSING, f"Invalid JSON response: {response.text[:100]}"),) * 4

        if not response_json.get("candidates"):
            feedback = response_json.get("promptFeedback")
            block_reason_msg = "ERROR: API_BLOCKED_UNKNOWN"
            if feedback and feedback.get("blockReason"):
                 block_reason = feedback.get("blockReason")
                 safety_ratings = feedback.get("safetyRatings", [])
                 logger.warning(f"Row {row_index}: API call blocked. Reason: {block_reason}. Ratings: {safety_ratings}")
                 block_reason_msg = create_error_message(ErrorType.API_CALL, f"BLOCKED_BY_API: {block_reason}")
            else:
                 logger.warning(f"Row {row_index}: No candidates in API response. Full response (truncated): {str(response_json)[:500]}")
                 block_reason_msg = create_error_message(ErrorType.API_CALL, "No candidates in response")
            return (block_reason_msg,) * 4

        candidate = response_json["candidates"][0]
        if not candidate.get("content") or not candidate["content"].get("parts"):
            finish_reason = candidate.get("finishReason")
            gen_stopped_msg = "ERROR: API_NO_CONTENT_PARTS"
            if finish_reason and finish_reason != "STOP":
                 logger.warning(f"Row {row_index}: Generation stopped by API. Reason: {finish_reason}")
                 gen_stopped_msg = create_error_message(ErrorType.API_CALL, f"GENERATION_STOPPED: {finish_reason}")
            else:
                 logger.warning(f"Row {row_index}: No content parts in API candidate. Candidate (truncated): {str(candidate)[:500]}")
            return (gen_stopped_msg,) * 4
        try:
            json_output_str = candidate["content"]["parts"][0]["text"]
            # Clean markdown code block indicators if present
            if json_output_str.startswith("```json"): json_output_str = json_output_str[7:]
            elif json_output_str.startswith("```"): json_output_str = json_output_str[3:]
            if json_output_str.endswith("```"): json_output_str = json_output_str[:-3]
            json_output_str = json_output_str.strip()

            parsed_json = json.loads(json_output_str)
            icebreaker = parsed_json.get("personalisedIcebreaker", "").strip()
            observation = parsed_json.get("specificObservation", "").strip()
            pain_point = parsed_json.get("specificPainPoint", "").strip()
            belief = parsed_json.get("impliedBelief", "").strip()

            missing_fields = []
            if not icebreaker: missing_fields.append("personalisedIcebreaker")
            if not observation: missing_fields.append("specificObservation")
            if not pain_point: missing_fields.append("specificPainPoint")
            if not belief: missing_fields.append("impliedBelief")

            if missing_fields:
                warn_msg = f"Row {row_index}: Parsed JSON but one or more expected keys were missing or empty: {', '.join(missing_fields)}. Raw JSON output: {json_output_str[:200]}"
                logger.warning(warn_msg)
                err_placeholder = create_error_message(ErrorType.JSON_PARSING, f"Missing keys ({', '.join(missing_fields)})")
                # Return what was found, and error for missing
                return (
                    icebreaker if icebreaker else err_placeholder,
                    observation if observation else err_placeholder,
                    pain_point if pain_point else err_placeholder,
                    belief if belief else err_placeholder
                )
            return icebreaker, observation, pain_point, belief
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e: # Added TypeError
            raw_text = candidate['content']['parts'][0].get('text', 'N/A') if candidate.get('content', {}).get('parts') else 'N/A'
            logger.error(f"Row {row_index}: Failed to parse or extract content from Gemini JSON. Error: {e}. Raw text: {raw_text[:500]}")
            err_msg = create_error_message(ErrorType.JSON_PARSING, f"Content parsing error: {str(e)[:50]}. Raw: {raw_text[:100]}")
            return (err_msg,) * 4
        except Exception as e: # Catch-all for unexpected errors during parsing
            logger.error(f"Row {row_index}: Unexpected error parsing Gemini response content: {e}", exc_info=True)
            return (create_error_message(ErrorType.API_CALL, "Unexpected response parsing error"),) * 4
    except requests.exceptions.Timeout as e:
        logger.error(f"Row {row_index}: API call timed out. Error: {e}")
        raise # Re-raise for backoff to handle
    except requests.exceptions.RequestException as e:
        logger.error(f"Row {row_index}: API call failed. Error: {e}")
        if e.response is not None:
            logger.error(f"Row {row_index}: Response status: {e.response.status_code}. Response text: {e.response.text[:500]}")
        raise # Re-raise for backoff to handle
    except Exception as e: # Catch-all for other unexpected errors
        logger.error(f"Row {row_index}: An unexpected error occurred in call_gemini_api: {e}", exc_info=True)
        final_err_msg = create_error_message(ErrorType.API_CALL, f"Unexpected API call failure: {str(e)[:100]}")
        return (final_err_msg,) * 4

def debug_api_call(api_key: str, url: str):
    logger.info("🔍 API Debug Information:")
    logger.info(f"  URL: {url}")
    logger.info(f"  API Key Present: {'Yes' if api_key else 'No'}")
    if api_key: logger.info(f"  API Key Suffix (last 4 chars): ...{api_key[-4:]}" if len(api_key) > 4 else "Too short")
    logger.info(f"  Model Name: {GEMINI_MODEL_NAME}")
    try:
        logger.info("  Testing API connectivity to list models...")
        response = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", timeout=10)
        if response.status_code == 200:
            available_models_data = response.json().get('models', [])
            available_model_names = [model.get('name') for model in available_models_data]
            logger.info(f"  API Connection: Success ✅ ({len(available_model_names)} models found)")
            logger.debug(f"  Available Models (first 5): {', '.join(available_model_names[:5])}{'...' if len(available_model_names) > 5 else ''}")
            model_full_name_pattern = f"models/{GEMINI_MODEL_NAME}"
            if model_full_name_pattern in available_model_names:
                logger.info(f"  Target Model '{GEMINI_MODEL_NAME}' found in available models.")
            else:
                logger.warning(f"  ⚠️ WARNING: Target Model '{GEMINI_MODEL_NAME}' (expected as {model_full_name_pattern}) NOT FOUND in list of available models! Check model name and API key permissions.")
        else:
            logger.error(f"  API Connection Error when listing models: {response.status_code} {response.text[:200]}")
    except Exception as e:
        logger.error(f"  API Connection Test Failed: {str(e)}")
    logger.info("API debug completed.")

def get_mock_api_response(index: int) -> Tuple[str, str, str, str]:
    return (f"Test icebreaker for row {index}", f"Test observation for row {index}",
            f"Test pain point for row {index}", f"Test belief for row {index}")

# --- Main Execution Logic ---
def main() -> None:
    start_time = datetime.now()
    logger.info("--- Starting Gemini Personalization Script ---")

    if TEST_MODE: logger.info(f"🔍 RUNNING IN TEST MODE - Will only process up to {MAX_TEST_ROWS} rows")
    if DRY_RUN: logger.info(f"🔍 DRY RUN MODE ENABLED - No actual API calls will be made")

    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY environment variable not set. Exiting.")
        return

    if not DRY_RUN:
        try:
            logger.info("Running API diagnostics...")
            debug_api_call(GOOGLE_API_KEY, GEMINI_API_URL)
        except Exception as e:
            logger.error(f"API diagnostics failed: {e}", exc_info=True)
            # Potentially exit if API diagnostics fail critical checks, or just warn

    try:
        check_dependencies()
        logger.info(f"Reading main input Excel file: {INPUT_FILE_PATH}")
        try:
            df = pd.read_excel(INPUT_FILE_PATH, sheet_name=INPUT_SHEET_NAME, engine='openpyxl')
        except FileNotFoundError:
            logger.error(f"Main input file not found: {INPUT_FILE_PATH}. Exiting.")
            return
        except Exception as e:
            logger.warning(f"Failed reading '{INPUT_SHEET_NAME}' with openpyxl ({e}). Trying default engine.")
            try:
                df = pd.read_excel(INPUT_FILE_PATH, sheet_name=INPUT_SHEET_NAME)
            except Exception as read_e:
                logger.error(f"Failed to read main Excel file {INPUT_FILE_PATH} (sheet: {INPUT_SHEET_NAME}): {read_e}. Exiting.")
                return

        logger.info(f"Successfully read {len(df)} rows from {INPUT_FILE_NAME} sheet {INPUT_SHEET_NAME}.")
        if df.empty:
            logger.warning("Main input file/sheet is empty. Nothing to process. Exiting.")
            return

        # --- LinkedIn Posts Data Integration ---
        df['formatted_linkedin_posts'] = "No specific recent LinkedIn posts provided for this contact.\n" # Default
        df['normalized_linkedin_url'] = "" # Initialize column

        logger.debug(f"Initial LI_POSTS_CONFIG: {LI_POSTS_CONFIG}")
        logger.debug(f"Initial LI_POSTS_FILE_NAME: {LI_POSTS_FILE_NAME}")
        logger.debug(f"Initial LINKEDIN_POSTS_FILE_PATH: {LINKEDIN_POSTS_FILE_PATH}")
        if LINKEDIN_POSTS_FILE_PATH:
            logger.debug(f"LinkedIn posts file exists check: {LINKEDIN_POSTS_FILE_PATH.exists()}")

        if LINKEDIN_POSTS_FILE_PATH and LI_POSTS_CONFIG and LINKEDIN_POSTS_FILE_PATH.exists():
            raw_main_df_key_col = LI_POSTS_CONFIG.get('key_column_main_df')
            raw_li_df_key_col = LI_POSTS_CONFIG.get('key_column_li_df')
            li_content_col = LI_POSTS_CONFIG.get('content_column')
            li_ts_col = LI_POSTS_CONFIG.get('timestamp_column')
            max_posts_per_person = LI_POSTS_CONFIG.get('max_posts_per_person', 2)
            max_post_age_days = LI_POSTS_CONFIG.get('max_post_age_days')

            if not all([raw_main_df_key_col, raw_li_df_key_col, li_content_col, li_ts_col]):
                logger.warning("LinkedIn posts configuration (key_column_main_df, key_column_li_df, content_column, timestamp_column) is incomplete in config.yaml. Skipping LinkedIn data integration.")
            elif raw_main_df_key_col not in df.columns:
                logger.warning(f"Key column '{raw_main_df_key_col}' for LinkedIn posts (from config: files.linkedin_posts.key_column_main_df) not found in main DataFrame. Skipping LinkedIn data integration.")
            else:
                try:
                    logger.info(f"Reading LinkedIn posts from: {LINKEDIN_POSTS_FILE_PATH}")
                    file_ext = LINKEDIN_POSTS_FILE_PATH.suffix.lower()
                    if file_ext == '.csv':
                        df_li_posts = pd.read_csv(LINKEDIN_POSTS_FILE_PATH, encoding='utf-8')
                    elif file_ext in ['.xlsx', '.xls']:
                        df_li_posts = pd.read_excel(LINKEDIN_POSTS_FILE_PATH, sheet_name=LI_POSTS_CONFIG.get('sheet', 0)) # Default to first sheet
                    else:
                        logger.error(f"Unsupported LinkedIn posts file extension: {file_ext}. Skipping LinkedIn data integration.")
                        df_li_posts = None

                    if df_li_posts is None or df_li_posts.empty:
                        logger.warning("No LinkedIn posts data loaded or file is empty. Skipping integration.")
                    elif not all(col in df_li_posts.columns for col in [raw_li_df_key_col, li_content_col, li_ts_col]):
                        logger.warning(f"LinkedIn posts file missing one or more required columns specified in config: '{raw_li_df_key_col}', '{li_content_col}', '{li_ts_col}'. Skipping LinkedIn data integration.")
                    else:
                        logger.info(f"Read {len(df_li_posts)} LinkedIn post entries.")

                        df[raw_main_df_key_col] = df[raw_main_df_key_col].astype(str)
                        df_li_posts[raw_li_df_key_col] = df_li_posts[raw_li_df_key_col].astype(str)

                        df['normalized_linkedin_url'] = df[raw_main_df_key_col].apply(sanitized_profile_url)
                        df_li_posts['normalized_linkedin_url'] = df_li_posts[raw_li_df_key_col].apply(sanitized_profile_url)

                        df = df[df['normalized_linkedin_url'].str.strip() != ''].copy() # Filter main df
                        df_li_posts = df_li_posts[df_li_posts['normalized_linkedin_url'].str.strip() != ''].copy() # Filter LI posts df

                        if df.empty:
                            logger.warning("Main DataFrame is empty after filtering out rows with invalid/empty LinkedIn URLs based on config. Skipping LinkedIn data integration.")
                        elif df_li_posts.empty:
                            logger.warning("LinkedIn posts DataFrame is empty after filtering out rows with invalid/empty profile URLs. Skipping LinkedIn data integration.")
                        else:
                            logger.debug(f"Example normalized main input profile URLs (first 5 unique): {list(df['normalized_linkedin_url'].unique())[:5]}")
                            logger.debug(f"Example normalized LinkedIn posts profile URLs (first 5 unique): {list(df_li_posts['normalized_linkedin_url'].unique())[:5]}")

                            df_li_posts[li_ts_col] = pd.to_datetime(df_li_posts[li_ts_col], errors='coerce')
                            df_li_posts.dropna(subset=[li_ts_col, li_content_col, 'normalized_linkedin_url'], inplace=True)

                            if max_post_age_days is not None:
                                cutoff_date = datetime.now() - timedelta(days=max_post_age_days)
                                # Handle timezones carefully for comparison
                                if pd.api.types.is_datetime64_any_dtype(df_li_posts[li_ts_col]) and df_li_posts[li_ts_col].dt.tz is not None:
                                    # If posts have timezone, make cutoff_date aware of the same timezone
                                    cutoff_date = cutoff_date.replace(tzinfo=df_li_posts[li_ts_col].dt.tz.zone) if hasattr(df_li_posts[li_ts_col].dt.tz, 'zone') else cutoff_date.astimezone(df_li_posts[li_ts_col].dt.tz)
                                else: # Posts are naive, make cutoff_date naive
                                    cutoff_date = cutoff_date.replace(tzinfo=None)
                                    if pd.api.types.is_datetime64_any_dtype(df_li_posts[li_ts_col]): # Ensure posts are naive if cutoff is naive
                                        df_li_posts[li_ts_col] = df_li_posts[li_ts_col].dt.tz_localize(None)
                                df_li_posts = df_li_posts[df_li_posts[li_ts_col] >= cutoff_date]
                                logger.info(f"Filtered to {len(df_li_posts)} LinkedIn posts within last {max_post_age_days} days.")

                            if df_li_posts.empty:
                                logger.info("No LinkedIn posts remained after date and content filtering. No posts will be merged.")
                            else:
                                df_li_posts.sort_values(by=['normalized_linkedin_url', li_ts_col], ascending=[True, False], inplace=True)

                                def format_posts_for_group(group_df: pd.DataFrame) -> str:
                                    formatted = []
                                    for _, post_row in group_df.head(max_posts_per_person).iterrows():
                                        age_str = format_linkedin_post_age(post_row[li_ts_col])
                                        content = safe_str(post_row[li_content_col])
                                        content_display = content[:300] + ('...' if len(content) > 300 else '')
                                        formatted.append(f"- Post ({age_str}): \"{content_display}\"")
                                    return "\n".join(formatted) if formatted else "No specific recent LinkedIn posts provided for this contact.\n"

                                aggregated_posts = df_li_posts.groupby('normalized_linkedin_url', as_index=False).apply(format_posts_for_group)
                                aggregated_posts.rename(columns={None: 'aggregated_formatted_posts'}, inplace=True) # Pandas 2.x+ might name it '' or 0
                                if '' in aggregated_posts.columns and 'aggregated_formatted_posts' not in aggregated_posts.columns : aggregated_posts.rename(columns={'': 'aggregated_formatted_posts'}, inplace=True)
                                elif 0 in aggregated_posts.columns and 'aggregated_formatted_posts' not in aggregated_posts.columns : aggregated_posts.rename(columns={0: 'aggregated_formatted_posts'}, inplace=True)


                                logger.info(f"Aggregated posts for {len(aggregated_posts)} unique normalized profiles from LinkedIn data.")

                                df = pd.merge(df, aggregated_posts, on='normalized_linkedin_url', how='left')
                                df['formatted_linkedin_posts'] = df['aggregated_formatted_posts'].fillna("No specific recent LinkedIn posts provided for this contact.\n")
                                df.drop(columns=['aggregated_formatted_posts'], inplace=True, errors='ignore')

                                matches_count = len(df[df['formatted_linkedin_posts'] != "No specific recent LinkedIn posts provided for this contact.\n"])
                                logger.info(f"Merged LinkedIn posts. {matches_count} prospects in main df have formatted LinkedIn posts. Main df length is {len(df)}.")

                                # Optional: Filter main DataFrame to only include rows that now have LI activity
                                # This was a previous explicit filter step:
                                # if not aggregated_posts.empty:
                                # profiles_with_aggregated_posts = set(aggregated_posts['normalized_linkedin_url'].unique())
                                # original_df_len = len(df)
                                # df = df[df['normalized_linkedin_url'].isin(profiles_with_aggregated_posts)].copy()
                                # logger.info(f"Filtered main DataFrame from {original_df_len} to {len(df)} rows, keeping only those whose profile URL matched an entry in the (date-filtered and aggregated) LinkedIn posts data.")
                                # else:
                                # logger.info("No posts were aggregated from LinkedIn file, so skipping filter of main DataFrame by LI profile matches.")

                                if not df.empty:
                                    first_with_posts = df[df['formatted_linkedin_posts'] != "No specific recent LinkedIn posts provided for this contact.\n"].head(1)
                                    if not first_with_posts.empty:
                                        logger.debug(f"Example of merged LinkedIn post data for profile '{first_with_posts['normalized_linkedin_url'].iloc[0]}':\n{first_with_posts['formatted_linkedin_posts'].iloc[0]}")
                                    else:
                                        logger.debug("No prospects had matching LinkedIn posts after merging.")
                                else:
                                    logger.warning("Main DataFrame is empty after LinkedIn post merging and any applied filters.")
                except FileNotFoundError: # Should be caught by .exists() check earlier
                    logger.warning(f"LinkedIn posts file not found at {LINKEDIN_POSTS_FILE_PATH}. Skipping LinkedIn data.")
                except Exception as e:
                    logger.error(f"Error processing LinkedIn posts data: {e}", exc_info=True)
        elif not LINKEDIN_POSTS_FILE_PATH or not LI_POSTS_CONFIG:
            logger.info("LinkedIn posts file not configured (path or config details missing). Skipping LinkedIn data integration.")
        elif LINKEDIN_POSTS_FILE_PATH and not LINKEDIN_POSTS_FILE_PATH.exists():
            logger.warning(f"LinkedIn posts file configured but not found at {LINKEDIN_POSTS_FILE_PATH}. Skipping LinkedIn data integration.")


        # If 'normalized_linkedin_url' wasn't created due to skipped LI processing, create it from main_df_key_col if possible
        if 'normalized_linkedin_url' not in df.columns or df['normalized_linkedin_url'].eq("").all():
            raw_main_df_key_col_fallback = LI_POSTS_CONFIG.get('key_column_main_df') if LI_POSTS_CONFIG else None
            if raw_main_df_key_col_fallback and raw_main_df_key_col_fallback in df.columns:
                logger.info(f"Populating 'normalized_linkedin_url' from '{raw_main_df_key_col_fallback}' as LI processing might have been skipped or yielded no keys.")
                df['normalized_linkedin_url'] = df[raw_main_df_key_col_fallback].astype(str).apply(sanitized_profile_url)
            else:
                logger.warning("Could not create/populate 'normalized_linkedin_url' as LinkedIn processing was skipped and 'key_column_main_df' is not available or configured.")


        required_cols_main = ["assignedPersona", "referenceableDataPoints", "first_name", "companyName",
                              "verifiedJobTitle", "individualActivitySummary", "companyNewsSummary"]
        missing_cols = [col for col in required_cols_main if col not in df.columns]
        if missing_cols:
            logger.error(f"Main input DataFrame is missing required columns: {', '.join(missing_cols)}. Exiting.")
            return

        logger.info("Validating referenceableDataPoints presence for rows to be processed...")
        initial_row_count_before_ref_filter = len(df)

        def is_valid_ref_data(val: Any) -> bool:
            if pd.isna(val): return False
            sval = str(val).strip().lower()
            # '[]', '{}' are considered "empty but valid" by parse_data_points,
            # but for this filter, we might want to exclude them if they mean "no actual data".
            # Let's assume '[]' or '{}' from source means "explicitly no points found" vs. actual data.
            # The prompt handles "None specific found from web search" if data_points list is empty.
            if sval in ('na', 'n/a', 'nan', 'none', 'null', '', '[]', '{}'):
                return False
            return True

        # Filter based on referenceableDataPoints OR having actual LinkedIn posts
        # A row is kept if it has valid web data OR valid LI data
        df_before_data_filter = len(df)
        df['has_web_data'] = df['referenceableDataPoints'].apply(is_valid_ref_data)
        df['has_li_data'] = df['formatted_linkedin_posts'].apply(lambda x: x != "No specific recent LinkedIn posts provided for this contact.\n")
        
        df = df[df['has_web_data'] | df['has_li_data']].copy()
        logger.info(f"Filtered DataFrame from {df_before_data_filter} to {len(df)} rows, keeping those with either valid 'referenceableDataPoints' OR actual LinkedIn post data.")

        if df.empty:
            logger.error("All rows filtered out due to missing both 'referenceableDataPoints' and usable LinkedIn post data. Nothing to process. Exiting.")
            return

        logger.info("Generating shortened company names...")
        df['shortenedCompanyName'] = df['companyName'].apply(get_shortened_company_name)
        if len(df) > 0 and logger.isEnabledFor(logging.DEBUG): # Log examples only if debug is on
            logger.debug("Example shortened names (first 3):")
            for i in range(min(3, len(df))):
                logger.debug(f"  Original: '{df['companyName'].iloc[i]}' -> Shortened: '{df['shortenedCompanyName'].iloc[i]}'")

        create_gemini_request_payload.fallback_count = 0 # Reset counter

        results_columns = ['generatedIcebreaker', 'generatedObservation', 'generatedPainPoint', 'generatedImpliedBelief', 'error']
        for col in results_columns:
            df[col] = "" if col == 'error' else "ERROR: Not Processed" # Initialize

        tasks_df = df # Full df after filtering
        if TEST_MODE and len(df) > MAX_TEST_ROWS:
            tasks_df = df.sample(n=min(MAX_TEST_ROWS, len(df)), random_state=42) # Ensure n <= len(df)
            logger.info(f"TEST MODE: Randomly selected {len(tasks_df)} of {len(df)} available rows for processing.")
        
        tasks_to_submit = list(tasks_df.iterrows())

        if not tasks_to_submit:
            logger.warning("No tasks to submit to Gemini (e.g., DataFrame empty after filtering or test mode selection). Exiting.")
            return

        processed_count, error_count, consecutive_errors = 0, 0, 0
        
        logger.info(f"Starting API calls with up to {MAX_WORKERS} workers for {len(tasks_to_submit)} tasks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {executor.submit(process_row_for_gemini, task, GOOGLE_API_KEY): task[0] for task in tasks_to_submit}
            
            try:
                for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(tasks_to_submit), desc="Generating Email Components"):
                    original_index = future_to_index[future]
                    try:
                        _, icebreaker, observation, pain_point, belief = future.result(timeout=FUTURE_TIMEOUT)
                        
                        df.loc[original_index, 'generatedIcebreaker'] = icebreaker
                        df.loc[original_index, 'generatedObservation'] = observation
                        df.loc[original_index, 'generatedPainPoint'] = pain_point
                        df.loc[original_index, 'generatedImpliedBelief'] = belief
                        
                        all_res = [icebreaker, observation, pain_point, belief]
                        current_row_has_error = any(isinstance(r, str) and r.startswith("ERROR_") for r in all_res)
                        
                        if current_row_has_error:
                            first_err_msg = next((r for r in all_res if isinstance(r, str) and r.startswith("ERROR_")), "ERROR_UNKNOWN")
                            df.loc[original_index, 'error'] = first_err_msg
                            error_count += 1
                            consecutive_errors += 1
                            logger.warning(f"Row index {original_index} resulted in error: {first_err_msg}")
                        else:
                            consecutive_errors = 0
                        
                        processed_count += 1 # Count as processed even if there's an error for rate calculation

                        if CONSECUTIVE_ERROR_THRESHOLD > 0 and consecutive_errors >= CONSECUTIVE_ERROR_THRESHOLD:
                            msg = f"STOPPING: {consecutive_errors} consecutive errors exceeded threshold of {CONSECUTIVE_ERROR_THRESHOLD}."
                            logger.error(msg); tqdm.write(msg)
                            raise RuntimeError(msg)
                        if processed_count > 10 and ERROR_PERCENTAGE_THRESHOLD > 0 and (error_count / processed_count * 100) > ERROR_PERCENTAGE_THRESHOLD:
                            err_rate = error_count / processed_count * 100
                            msg = f"STOPPING: Error rate {err_rate:.1f}% exceeded threshold of {ERROR_PERCENTAGE_THRESHOLD}% after {processed_count} processed rows."
                            logger.error(msg); tqdm.write(msg)
                            raise RuntimeError(msg)
                    
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Row index {original_index} timed out after {FUTURE_TIMEOUT}s.")
                        error_msg = create_error_message(ErrorType.FUTURE_EXCEPTION, "Task timed out")
                        for col in results_columns[:-1]: df.loc[original_index, col] = error_msg # All content fields
                        df.loc[original_index, 'error'] = error_msg
                        error_count +=1; consecutive_errors +=1; processed_count +=1
                    except Exception as exc:
                        logger.error(f'Row index {original_index} generated an unhandled exception in main loop: {exc}', exc_info=True)
                        error_msg = create_error_message(ErrorType.FUTURE_EXCEPTION, str(exc))
                        for col in results_columns[:-1]: df.loc[original_index, col] = error_msg # All content fields
                        df.loc[original_index, 'error'] = error_msg
                        error_count += 1; consecutive_errors += 1; processed_count +=1
            
            except (KeyboardInterrupt, RuntimeError) as e:
                if isinstance(e, KeyboardInterrupt): logger.warning("Process interrupted by user.")
                else: logger.error(f"Processing stopped early: {e}")
                logger.warning("Cancelling remaining tasks and saving partial results...")
                for fut_to_cancel in future_to_index.keys(): # Iterate over original keys
                    if not fut_to_cancel.done():
                        fut_to_cancel.cancel()
                # Allow already completed futures to be processed by the loop if it wasn't a hard exit
            
        if hasattr(create_gemini_request_payload, 'fallback_count') and create_gemini_request_payload.fallback_count > 0:
            logger.info(f"Persona guidance fallback to 'General Professional' used {create_gemini_request_payload.fallback_count} times for {processed_count} processed rows.")
        
        logger.info(f"Finished processing loop. API calls attempted for {processed_count}/{len(tasks_to_submit)} selected tasks.")
        if error_count > 0 and processed_count > 0:
            err_rate_final = error_count / processed_count * 100
            logger.warning(f"Encountered errors in {error_count} rows ({err_rate_final:.1f}% of processed). Check 'error' column in output.")
        elif error_count > 0 and processed_count == 0:
             logger.warning(f"Encountered {error_count} errors, but no rows were fully processed (likely due to early termination or all tasks failing).")


        # Prepare DataFrame for saving - use tasks_df if in TEST_MODE and sampled, else full df
        df_to_save = tasks_df if (TEST_MODE and len(df) > MAX_TEST_ROWS and len(tasks_df) < len(df)) else df
        
        # Filter df_to_save to include only rows that were actually part of tasks_to_submit (i.e., had an attempt)
        # This handles cases where TEST_MODE didn't sample, but processing might have stopped early.
        # The results are already mapped to the correct indices in the original `df`.
        # We need to select the subset of `df` that corresponds to `tasks_to_submit`.
        processed_indices = [task[0] for task in tasks_to_submit] # These are the indices that were submitted
        df_to_save = df.loc[df.index.isin(processed_indices)].copy()


        if not df_to_save.empty:
            df_to_save['processedTimestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_to_save['geminiModelUsed'] = GEMINI_MODEL_NAME
            # Drop temporary helper columns if they exist
            df_to_save.drop(columns=['has_web_data', 'has_li_data'], inplace=True, errors='ignore')
            # Ensure 'shortenedCompanyName' is present in the output
            if 'shortenedCompanyName' not in df_to_save.columns:
                df_to_save['shortenedCompanyName'] = df['shortenedCompanyName']
        else:
            logger.warning("No data to save as df_to_save is empty. This might happen if input was empty, all rows filtered out, or all tasks failed/cancelled very early.")

        output_path = OUTPUT_FILE_PATH
        if TEST_MODE:
            base_name, ext = os.path.splitext(OUTPUT_FILE_NAME)
            output_path = OUTPUT_DIR / f"{base_name}_TEST_MODE{ext}"
            logger.info(f"Test mode: Saving results to {output_path}...")
        else:
            logger.info(f"Saving results to {output_path}...")

        if not df_to_save.empty:
            try:
                df_to_save.to_excel(output_path, index=False, engine='openpyxl')
                logger.info(f"Results saved: {len(df_to_save)} rows to {output_path}.")
                if len(df_to_save) > 0: # Redundant check, but safe
                    error_rows_in_saved_file = df_to_save[df_to_save['error'] != ""]
                    error_rate_in_saved_file = len(error_rows_in_saved_file) / len(df_to_save) * 100 if len(df_to_save) > 0 else 0
                    logger.info(f"Saved data summary: {len(error_rows_in_saved_file)} rows with errors ({error_rate_in_saved_file:.1f}%).")
            except Exception as e:
                logger.error(f"Failed to save output file to {output_path}: {e}", exc_info=True)
        else:
            logger.warning("No results to save as the final processed DataFrame (df_to_save) is empty.")

    except FileNotFoundError as e: # Should be caught earlier for main input
        logger.error(f"A critical file was not found: {e}", exc_info=True)
    except ImportError as e:
        logger.error(f"Import Error: {e}", exc_info=True) # Should be caught by check_dependencies
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main execution block: {e}", exc_info=True)
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"--- Script finished in {duration} ---")

if __name__ == '__main__':
    main()