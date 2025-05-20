import pandas as pd
import json
import os
import time
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv
from tqdm import tqdm
import random
import re # Import regex library for URL cleaning

# --- Configuration ---
load_dotenv()

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Load configuration
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"ERROR: Configuration file not found at {CONFIG_PATH}")
    exit(1)
except Exception as e:
    print(f"ERROR: Failed to load configuration: {e}")
    exit(1)

# Set up logging early, before any functions that use it
LOG_FILE_PATH = SCRIPT_DIR / config["paths"]["output"]["log"]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w'), # Overwrite log file each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI Client
API_KEY = os.getenv(config["openai"]["api_key_env"])
if not API_KEY:
    raise ValueError(f"{config['openai']['api_key_env']} environment variable not set.")

client = OpenAI(api_key=API_KEY)

# --- Function to resolve input file path ---
def resolve_input_file_path(base_path: Path, config_path: str, file_type: str) -> Tuple[Path, str]:
    """
    Resolves input file path checking both Excel and CSV formats.
    Uses the new folder structure with input/posts/ and input/websearch/ subfolders.
    Returns a tuple of (resolved_path, file_format)
    """
    # Try the configured path first
    config_file_path = base_path / config_path
    if config_file_path.exists():
        if config_file_path.suffix.lower() in ['.xlsx', '.xls']:
            return config_file_path, 'excel'
        elif config_file_path.suffix.lower() == '.csv':
            return config_file_path, 'csv'
    
    # Determine the appropriate subfolder based on file type
    if file_type == 'profiles':
        subfolder = base_path / "input" / "websearch"
    elif file_type == 'posts':
        subfolder = base_path / "input" / "posts"
    else:
        subfolder = config_file_path.parent
    
    # Ensure the subfolder exists
    if not subfolder.exists():
        logger.warning(f"Subfolder {subfolder} does not exist")
        return config_file_path, 'unknown'
    
    # For profile data, look for Excel files in websearch folder
    if file_type == 'profiles':
        # Look for any Excel files in the websearch folder
        excel_files = list(subfolder.glob("*.xlsx"))
        if excel_files:
            logger.info(f"Found {len(excel_files)} Excel files in {subfolder}. Using: {excel_files[0].name}")
            return excel_files[0], 'excel'
    
    # For posts data, look for "result (XX).csv" files in posts folder
    if file_type == 'posts':
        # Look for any "result (XX).csv" files
        result_files = list(subfolder.glob("result (*).csv"))
        if result_files:
            # Sort by the number in parentheses to get the latest one
            result_files.sort(key=lambda x: int(re.search(r'result \((\d+)\)\.csv', x.name).group(1)), reverse=True)
            logger.info(f"Found {len(result_files)} 'result (XX).csv' files in {subfolder}. Using the most recent: {result_files[0].name}")
            return result_files[0], 'csv'
    
    # Return the original path as fallback
    return config_file_path, 'unknown'

# --- NEW: Function to load DataFrame from file based on format ---
def load_dataframe(file_path: Path, file_format: str, sheet_name: Optional[str] = None, dtype: Optional[Dict] = None) -> pd.DataFrame:
    """
    Loads a DataFrame from a file based on its format.
    """
    if file_format == 'excel':
        return pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl', dtype=dtype)
    elif file_format == 'csv':
        # For CSV files we don't use sheet_name
        return pd.read_csv(file_path, dtype=dtype)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

# --- File Paths ---
INPUT_PROFILES_PATH, PROFILES_FORMAT = resolve_input_file_path(
    SCRIPT_DIR, 
    config["paths"]["input"]["excel"],
    'profiles'
)

# --- Posts Data ---
INPUT_POSTS_PATH, POSTS_FORMAT = resolve_input_file_path(
    SCRIPT_DIR, 
    config["paths"]["input"]["posts_excel"],
    'posts'
)

INPUT_SHEET_NAME = config["paths"]["input"]["sheet_name"]
OUTPUT_EXCEL_PATH = SCRIPT_DIR / config["paths"]["output"]["excel"]

# Ensure output directory exists
OUTPUT_EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Persona Definitions from Config ---
# (Keep your existing PERSONA_DEFINITIONS loading logic here)
PERSONA_DEFINITIONS = {
    "Problem/Goal Orientation": {
        persona: data["description"]
        for persona, data in config["personas"]["problem_goal_orientation"].items()
    },
    "Communication Focus": {
        persona: data["description"]
        for persona, data in config["personas"]["communication_focus"].items()
    },
    "Role-Based Archetype": {
        persona: data["description"]
        for persona, data in config["personas"]["role_based_archetype"].items()
    },
    "Default/Uncategorized": {
        persona: data["description"]
        for persona, data in config["personas"]["default"].items()
    }
}

# --- Helper function for URL/Email Sanitization ---
def sanitize_linkedin_url(value: Optional[str]) -> Optional[str]:
    """Extracts domain from email or cleans LinkedIn URL for consistent matching."""
    if pd.isna(value) or not isinstance(value, str):
        return None
    
    # Handle email addresses
    if '@' in value:
        # Extract domain from email (everything after @)
        domain = value.split('@')[-1].lower()
        return domain
    
    # Handle LinkedIn URLs
    if 'linkedin.com' in value.lower():
        # Extract the unique identifier part of the URL
        url = value.lower()
        
        # Remove any trailing parameters or commas and their content
        url = url.split(',')[0]
        url = url.split('?')[0]
        
        # Extract the unique identifier - it's the last part of the path
        parts = url.rstrip('/').split('/')
        unique_id = parts[-1]
        
        # For sales URLs, we want to preserve the unique ID but standardize the format
        if 'sales/lead' in url:
            return f"linkedin.com/in/{unique_id}"
        
        return f"linkedin.com/in/{unique_id}"
    
    return value.lower()  # Return normalized value for other cases

def format_persona_definitions_for_prompt(definitions: Dict) -> str:
    """Formats the persona definitions for inclusion in the AI prompt."""
    # (Keep your existing function code here)
    prompt_text = "PERSONA DEFINITIONS:\n\n"
    for category, personas in definitions.items():
        prompt_text += f"## {category}\n"
        for name, desc in personas.items():
            prompt_text += f"- **{name}:** {desc}\n"
        prompt_text += "\n"
    return prompt_text


def get_persona_category(data: Dict[str, Any], persona_definitions_text: str) -> Dict[str, Optional[str]]:
    """
    Calls the OpenAI API to categorize a prospect based on provided data (including posts) and persona definitions.
    """
    # --- MODIFIED: Include recentPostsContent and adjust required fields if needed ---
    required_fields = [
        "verifiedJobTitle", "inferredSeniority", "companyIndustry",
        "inferredCompanyStage", "individualActivitySummary", "companyNewsSummary",
        "referenceableDataPoints_json", "title", "summary", "titleDescription",
        "recentPostsContent" # Add the new field
    ]
    # --- END MODIFIED ---

    input_context = "\nPROSPECT DATA:\n"
    for field in required_fields:
        value = data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            value_str = "Not Available / Empty"
        elif field == "referenceableDataPoints_json":
            try:
                # Ensure value is treated as a string before loading
                json_str = str(value) if not isinstance(value, str) else value
                if not json_str or json_str.lower() == 'nan':
                    value_str = "None Found"
                else:
                    points = json.loads(json_str)
                    if points:
                        value_str = "; ".join([p.get('detail', 'N/A') for p in points])
                    else:
                        value_str = "None Found"
            except (json.JSONDecodeError, TypeError):
                 value_str = f"Invalid Data Points Format ({type(value)}): {str(value)[:100]}" # Log type and preview
        # --- NEW: Handle formatting for the new posts content ---
        elif field == "recentPostsContent":
            if not value or pd.isna(value):
                 value_str = "No recent posts available."
            else:
                 # Add truncation if posts are very long to avoid excessive prompt length
                 max_post_length = 1500 # Adjust as needed
                 value_str = str(value)[:max_post_length]
                 if len(str(value)) > max_post_length:
                     value_str += "... [truncated]"
        # --- END NEW ---
        else:
            value_str = str(value)
        input_context += f"- {field}: {value_str}\n"

    # --- MODIFIED: Update the prompt to mention post content ---
    prompt = f"""{persona_definitions_text}
{input_context}

**TASK:**
Based ONLY on the provided PROSPECT DATA (including Profile info and Recent Posts Content) and the PERSONA DEFINITIONS, assign the single best-fitting persona category to this prospect. Follow this prioritization logic:

1.  **Problem/Goal Orientation:** If the Company News Summary *or* Recent Posts Content strongly suggests a current major initiative (like scaling after funding, cost-cutting drive, major launch, risk mitigation focus) that aligns with the prospect's likely responsibilities (based on title/seniority), prioritize assigning a persona from the 'Problem/Goal Orientation' category.
2.  **Communication Focus:** If the Problem/Goal fit isn't strong, but the Individual Activity Summary *or the tone/topics in Recent Posts Content* reveals distinct themes or a clear communication style (e.g., focus on innovation, practical solutions, business results), prioritize assigning a persona from the 'Communication Focus' category.
3.  **Role-Based Archetype:** If neither of the above provides a strong signal, assign the most appropriate persona from the 'Role-Based Archetype' category based on the Title, Seniority, and Industry.
4.  **Default:** If the data is too sparse, conflicting, or doesn't clearly fit any specific persona, assign 'General Professional'.

**OUTPUT FORMAT:**
Return a single, valid JSON object with exactly two keys:
1.  `assignedPersona`: The name (string) of the single best-fitting persona category you selected.
2.  `reasoning`: A brief explanation (string, 1-2 sentences) justifying your choice, referencing specific points from the PROSPECT DATA (mentioning profile data or post content where relevant) and the prioritization logic.

Example Output (considering posts):
{{
  "assignedPersona": "The Innovator",
  "reasoning": "VP Engineering role combined with recent posts consistently focusing on bleeding-edge tech and experimental projects suggests an 'Innovator' focus, overriding general company stage."
}}

Provide only the JSON object in your response.
"""
    # --- END MODIFIED ---

    retries = 0
    while retries < config["openai"]["max_retries"]:
        try:
            response = client.chat.completions.create(
                model=config["openai"]["model"],
                messages=[
                    {"role": "system", "content": "You are an expert analyst classifying professionals into predefined persona categories based on structured data (profile and recent posts) and specific prioritization rules. You output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config["openai"]["temperature"],
                max_tokens=config["openai"]["max_tokens"],
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning(f"Received empty content for prospect URL: {data.get('linkedInProfileUrl', 'N/A')}") # Use a consistent ID if available
                return {"assignedPersona": "Error: Empty Response", "reasoning": None}

            try:
                result_json = json.loads(content)
                if 'assignedPersona' in result_json and 'reasoning' in result_json:
                    return {
                        "assignedPersona": result_json.get('assignedPersona'),
                        "reasoning": result_json.get('reasoning')
                    }
                else:
                    logger.warning(f"Response missing required keys for prospect URL: {data.get('linkedInProfileUrl', 'N/A')}. Response: {content}")
                    return {"assignedPersona": "Error: Invalid JSON Structure", "reasoning": content}

            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON response for prospect URL: {data.get('linkedInProfileUrl', 'N/A')}. Response: {content}")
                retries += 1
                if retries < config["openai"]["max_retries"]:
                    time.sleep(config["openai"]["retry_delay_seconds"] * (2 ** retries) + random.uniform(0, 1))
                else:
                    return {"assignedPersona": "Error: JSON Decode Failed", "reasoning": content}

        except RateLimitError as e:
            retries += 1
            logger.warning(f"Rate limit hit, retrying ({retries}/{config['openai']['max_retries']}) for prospect URL: {data.get('linkedInProfileUrl', 'N/A')}. Error: {e}")
            if retries < config["openai"]["max_retries"]:
                time.sleep(config["openai"]["retry_delay_seconds"] * (2 ** retries) + random.uniform(0, 1))
            else:
                logger.error(f"Max retries exceeded for rate limit on prospect URL: {data.get('linkedInProfileUrl', 'N/A')}.")
                return {"assignedPersona": "Error: Rate Limited", "reasoning": str(e)}
        except (APIError, APIConnectionError) as e:
            retries += 1
            logger.warning(f"API error, retrying ({retries}/{config['openai']['max_retries']}) for prospect URL: {data.get('linkedInProfileUrl', 'N/A')}. Error: {e}")
            if retries < config["openai"]["max_retries"]:
                time.sleep(config["openai"]["retry_delay_seconds"] * (2 ** retries) + random.uniform(0, 1))
            else:
                logger.error(f"Max retries exceeded for API error on prospect URL: {data.get('linkedInProfileUrl', 'N/A')}.")
                return {"assignedPersona": "Error: API Failed", "reasoning": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during API call for prospect URL: {data.get('linkedInProfileUrl', 'N/A')}: {e}", exc_info=True)
            return {"assignedPersona": "Error: Unexpected", "reasoning": str(e)}

    logger.error(f"Max retries exceeded after all attempts for prospect URL: {data.get('linkedInProfileUrl', 'N/A')}.")
    return {"assignedPersona": "Error: Max Retries Exceeded", "reasoning": None}


def main():
    """Main function to orchestrate persona categorization."""
    logger.info("--- Starting Persona Categorization Script ---")

    # --- Validate Inputs ---
    if not INPUT_PROFILES_PATH.is_file():
        logger.critical(f"Input Profile file not found: {INPUT_PROFILES_PATH}")
        return
    
    logger.info(f"Using {PROFILES_FORMAT.upper()} format for profile data: {INPUT_PROFILES_PATH}")
    
    # --- Validate Posts File ---
    if not INPUT_POSTS_PATH.is_file():
        logger.warning(f"Input Posts file not found: {INPUT_POSTS_PATH}. Proceeding without post data.")
        df_posts = None # Indicate that posts data is unavailable
    else:
        logger.info(f"Found Posts data file ({POSTS_FORMAT.upper()} format): {INPUT_POSTS_PATH}")

    if not API_KEY:
        logger.critical("OpenAI API Key not configured.")
        return

    # --- Load Profile Data ---
    logger.info(f"Reading profile data from: {INPUT_PROFILES_PATH}")
    try:
        # Important: Ensure the profile URL is read as a string
        df_profiles = load_dataframe(
            INPUT_PROFILES_PATH, 
            PROFILES_FORMAT,
            sheet_name=INPUT_SHEET_NAME if PROFILES_FORMAT == 'excel' else None,
            dtype={'linkedInProfileUrl': str}
        )
        logger.info(f"Successfully loaded {len(df_profiles)} profile rows.")
    except Exception as e:
        logger.critical(f"Failed to read input Profile file: {e}")
        return

    # --- Identify Profile URL Column ---
    # Adjust this if your profile URL column name is different
    profile_url_col_name = 'linkedInProfileUrl'
    if profile_url_col_name not in df_profiles.columns:
         alt_names = ['profileUrl', 'linkedinProfileUrl', 'LinkedInProfileURL', 'Email'] # Added Email as alternative
         found = False
         for name in alt_names:
             if name in df_profiles.columns:
                 profile_url_col_name = name
                 logger.info(f"Using profile URL column: '{profile_url_col_name}'")
                 found = True
                 break
         if not found:
             logger.critical(f"Could not find a suitable LinkedIn Profile URL column in {INPUT_PROFILES_PATH}. Searched for '{profile_url_col_name}' and alternatives.")
             return

    # --- Sanitize URLs in Profile Data FIRST ---
    logger.info(f"Sanitizing URLs in profile data '{profile_url_col_name}' column...")
    df_profiles['sanitized_url'] = df_profiles[profile_url_col_name].apply(sanitize_linkedin_url)
    invalid_profile_urls = df_profiles['sanitized_url'].isna().sum()
    if invalid_profile_urls > 0:
        logger.warning(f"{invalid_profile_urls} profiles have missing or invalid URLs after sanitization.")

    # --- Check required columns in profile data ---
    required_profile_columns = [
        "verifiedJobTitle", "inferredSeniority", "companyIndustry",
        "inferredCompanyStage", "individualActivitySummary", "companyNewsSummary",
        "referenceableDataPoints_json", "title", "summary", "titleDescription",
        profile_url_col_name
    ]
    missing_cols = [col for col in required_profile_columns if col not in df_profiles.columns]
    if missing_cols:
        logger.warning(f"Input data is missing some recommended profile columns: {', '.join(missing_cols)}")
        logger.info("Script will continue with available data.")

    # --- Load and Process Posts Data ---
    aggregated_posts = None
    if INPUT_POSTS_PATH.is_file():
        try:
            logger.info(f"Reading posts data from: {INPUT_POSTS_PATH}")
            # Ensure profileUrl is read as string
            df_posts = load_dataframe(
                INPUT_POSTS_PATH, 
                POSTS_FORMAT,
                dtype={'profileUrl': str}
            )
            logger.info(f"Successfully loaded {len(df_posts)} post rows.")

            # --- Identify Post URL Column ---
            posts_url_col_name = 'profileUrl'
            if posts_url_col_name not in df_posts.columns:
                logger.error(f"Posts file {INPUT_POSTS_PATH} is missing the required '{posts_url_col_name}' column. Cannot link posts.")
                df_posts = None
            elif 'postContent' not in df_posts.columns:
                logger.error(f"Posts file {INPUT_POSTS_PATH} is missing the required 'postContent' column. Cannot use posts.")
                df_posts = None
            else:
                # Drop rows where essential columns are missing
                df_posts = df_posts.dropna(subset=[posts_url_col_name, 'postContent'])
                if df_posts.empty:
                    logger.warning("No valid post entries found after removing rows with missing URL or content.")
                    df_posts = None
                else:
                    # --- Sanitize URLs in Posts Data ---
                    logger.info(f"Sanitizing URLs in posts data '{posts_url_col_name}' column...")
                    df_posts['sanitized_url'] = df_posts[posts_url_col_name].apply(sanitize_linkedin_url)
                    df_posts = df_posts.dropna(subset=['sanitized_url'])
                    
                    # Get unique sanitized URLs from profile data
                    profile_urls = set(df_profiles['sanitized_url'].dropna().unique())
                    logger.info(f"Found {len(profile_urls)} unique profile URLs to match against.")
                    
                    # Debug logging for URL matching
                    logger.info(f"Sample of profile URLs: {list(profile_urls)[:5]}")
                    logger.info(f"Sample of post URLs: {df_posts['sanitized_url'].unique()[:5]}")
                    
                    # Filter posts to only include those matching profile URLs
                    df_posts = df_posts[df_posts['sanitized_url'].isin(profile_urls)]
                    
                    # Early validation: Check if we have enough matching profiles
                    matching_profiles = len(df_posts['sanitized_url'].unique())
                    if matching_profiles < 5:
                        logger.critical(f"Critical Error: Only found {matching_profiles} matching profiles between files.")
                        logger.critical("This suggests a potential mismatch between the files or incorrect URL/email formats.")
                        logger.critical("Please verify that:")
                        logger.critical("1. Both files contain the correct identifiers (emails or LinkedIn URLs)")
                        logger.critical("2. The identifiers are in the same format across both files")
                        logger.critical("3. The files are for the same set of profiles")
                        return
                    
                    if df_posts.empty:
                        logger.warning("No matching posts found for any profiles. Proceeding without post data.")
                        logger.info("Please check that the email domains or LinkedIn URLs match between files.")
                        df_posts = None
                    else:
                        matching_count = len(df_posts['sanitized_url'].unique())
                        logger.info(f"Found {len(df_posts)} post entries matching {matching_count} unique profiles.")
                        
                        # Sort by timestamp within each group before aggregating
                        if 'postTimestamp' in df_posts.columns:
                            df_posts['postTimestamp'] = pd.to_datetime(df_posts['postTimestamp'], errors='coerce')
                            df_posts = df_posts.sort_values(['sanitized_url', 'postTimestamp'], ascending=[True, False])

                        # Group by sanitized URL and join post content
                        aggregated_posts = df_posts.groupby('sanitized_url')['postContent'].apply(
                            lambda posts: "\n---\n".join(posts.astype(str).tolist())
                        ).reset_index()
                        aggregated_posts.rename(columns={'postContent': 'recentPostsContent'}, inplace=True)
                        logger.info(f"Successfully aggregated posts for {len(aggregated_posts)} unique profiles.")

        except Exception as e:
            logger.error(f"Failed to read or process posts file: {e}", exc_info=True)
            aggregated_posts = None

    # --- Merge Posts Data into Profile Data ---
    if aggregated_posts is not None and not aggregated_posts.empty:
        logger.info("Merging aggregated post data with profile data...")
        # Debug logging before merge
        logger.info(f"Profile data shape before merge: {df_profiles.shape}")
        logger.info(f"Aggregated posts shape: {aggregated_posts.shape}")
        
        # Use left merge to keep all profiles, adding posts where available
        df_merged = pd.merge(
            df_profiles,
            aggregated_posts,
            on='sanitized_url',
            how='left'
        )
        
        # Debug logging after merge
        logger.info(f"Merged data shape: {df_merged.shape}")
        posts_count = df_merged['recentPostsContent'].notna().sum()
        logger.info(f"Profiles with posts after merge: {posts_count}")
        
        # Fill missing post content with a placeholder
        df_merged['recentPostsContent'].fillna("No recent posts found or matched.", inplace=True)
        logger.info(f"Merge completed. {len(df_merged)} total profiles, {posts_count} with post data.")
    else:
        logger.info("No aggregated post data available to merge. Proceeding with profile data only.")
        df_merged = df_profiles
        df_merged['recentPostsContent'] = "No recent posts available."

    # Keep the original profile URL for reference if needed, remove sanitized helper column if desired
    # df_merged = df_merged.drop(columns=['sanitized_url']) # Optional cleanup

    # --- Prepare Persona Definitions for Prompt ---
    persona_definitions_text = format_persona_definitions_for_prompt(PERSONA_DEFINITIONS)

    # --- Process Rows (using the merged dataframe) ---
    results_personas = []
    results_reasoning = []

    logger.info(f"Starting categorization using model: {config['openai']['model']} with profile and post data.")
    # Use df_merged which now contains profile + post data
    for index, row in tqdm(df_merged.iterrows(), total=df_merged.shape[0], desc="Categorizing Prospects"):
        row_data = row.to_dict()
        categorization_result = get_persona_category(row_data, persona_definitions_text)
        results_personas.append(categorization_result.get("assignedPersona"))
        results_reasoning.append(categorization_result.get("reasoning"))

    # --- Add results to DataFrame ---
    df_merged['assignedPersona'] = results_personas
    df_merged['personaReasoning'] = results_reasoning
    logger.info("Finished processing all rows.")

    # --- Save Output ---
    logger.info(f"Saving categorized results to: {OUTPUT_EXCEL_PATH}")
    try:
        # Save the merged dataframe with the new columns
        df_merged.to_excel(OUTPUT_EXCEL_PATH, index=False, engine='openpyxl')
        logger.info("Results saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save output Excel file: {e}")

    logger.info("--- Persona Categorization Script Finished ---")

if __name__ == "__main__":
    main()