# Leads Rapidly Outreach Process

This project is designed to automate and streamline the outreach process for leads, using a series of data processing, enrichment, persona mapping, and icebreaker generation steps. The workflow is intended to be run sequentially as follows:

1. **Preprocess**
2. **Websearch**
3. **Persona Mapping**
4. **Icebreaker**

Below is an overview of each main subfolder and its role in the process.

---

## 1. `leads_rapidly_preprocess/`

**Purpose:**  
Initial data preparation and merging of lead information, including LinkedIn profile scraping.

**Key contents:**
- `leads_rapidly_outreach_pre-process/`: Contains scripts and data for merging and cleaning lead lists.
  - `merge_leads_w_scrape.py`: Script to merge lead data with LinkedIn profile scrape results.
  - `Ross_leads_rapidly_SW_1450_CLEAN.csv`, `Ross_leads_rapidly_SW_1450.csv`, `Ross_leads_rapidly_SW_1450_LI_URL.csv`: Various stages of the lead data.
  - `pb_LI_profile_scrape/`: Contains LinkedIn profile scrape results (e.g., `result (20).csv`).

---

## 2. `leads_rapidly_websearch/`

**Purpose:**  
Enriches leads with additional information from web searches and processes batches of leads.

**Key contents:**
- `input/`: Contains the main input CSV for this stage (e.g., `Ross_leads_rapidly_SW_1450_CLEAN.csv`).
- `output/`: Stores the results of websearch enrichment, organised by run and batch.
- `leads_rapidly_outreach_pre-process/`: May contain additional scripts or data for pre-processing.
- Main scripts:  
  - `leads_rapidly_batch_to_openai_responses.py`, `batch_recovery_handler.py`, `batch_id_recovery.py`, `batch_debugging.py`: Scripts for batch processing, error recovery, and debugging.
- Configuration:  
  - `config.yaml`: Configuration file for this stage.
- Logs:  
  - Various `.log` files for tracking processing and errors.

---

## 3. `leads_rapidly_persona_mapping/`

**Purpose:**  
Maps enriched leads to personas based on their web and post activity.

**Key contents:**
- `input/`: Contains subfolders for `websearch` and `posts` data used in persona mapping.
- `output/`: Stores the resulting persona mapping, e.g., `categorized_personas.xlsx`.
- Main script:  
  - `main.py`: The primary script for persona categorisation.
- Configuration:  
  - `config.yaml`: Configuration file for persona mapping.
- Logs:  
  - `persona_categorization.log`: Log file for this stage.

---

## 4. `leads_rapidly_icebreaker/`

**Purpose:**  
Generates personalised icebreakers for each lead based on their persona and activity.

**Key contents:**
- `input/`: Contains input data for icebreaker generation, including:
  - `LI_post_activity/`: Likely contains LinkedIn post activity data.
  - `categorized_personas.xlsx`: Output from the previous step, used as input here.
- `output/`: Stores the generated icebreakers, e.g., `processed_leads.xlsx`.
- Main script:  
  - `main.py`: The primary script for generating icebreakers.
- Configuration:  
  - `config.yaml`: Configuration file for this stage.
- Logs:  
  - `icebreaker_generation.log`: Log file for this stage.

---

## Running the Pipeline

**Recommended order:**
1. Run the scripts in `leads_rapidly_preprocess` to prepare and clean your lead data.
2. Use `leads_rapidly_websearch` to enrich the leads with additional information.
3. Map personas using the scripts in `leads_rapidly_persona_mapping`.
4. Generate icebreakers with `leads_rapidly_icebreaker`.

Each stage produces outputs that are used as inputs for the next stage. Please refer to the individual `README.md` files within each subfolder for detailed instructions and requirements.

---

If you need further details or wish to automate the entire pipeline, consider creating a master script or workflow manager to run each stage in sequence. 