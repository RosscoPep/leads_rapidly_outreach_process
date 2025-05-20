# Gemini Batch Processing

A Python script for batch processing LinkedIn profiles using Google's Gemini AI model to generate personalised icebreakers and use cases for cold emails.

## Overview

This script processes two main data sources:
1. A personas file containing LinkedIn profile and company data
2. A LinkedIn activity file containing recent posts

It uses the Gemini AI model to generate:
- Personalised icebreakers based on recent LinkedIn activity and web reference points
- Relevant use cases and tailored email components based on the prospect's role, persona, and company context

The script is designed for robust data validation, efficient processing, and detailed logging.

## Features

- Batch processing of LinkedIn profiles from Excel and CSV files
- Robust matching of profiles using normalised LinkedIn URLs
- Only processes rows with valid LinkedIn activity or referenceable data points
- Integration with Google's Gemini AI model
- Rate limiting and retry logic with exponential backoff
- Concurrent processing with configurable worker count
- Comprehensive logging and error handling
- Data validation and integrity checks
- Test mode and dry run support via config
- Output includes a 'shortenedCompanyName' column for each row
- UTF-8 encoding support for international characters

## Prerequisites

- Python 3.8+
- Google API Key with access to Gemini AI
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RosscoPep/gemini-batch-processing.git
cd gemini-batch-processing
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemini-2.0-flash
GEMINI_RPM=15
MAX_WORKERS=1
```

## Usage

1. Prepare your input files as specified in `config.yaml`:
   - The personas file (e.g. `input/categorized_personas.xlsx`) with columns such as `first_name`, `companyName`, `assignedPersona`, `referenceableDataPoints`, and `sanitized_url` or `linkedinProfileUrl`.
   - The LinkedIn activity file (e.g. `input/LI_post_activity/result (20).csv`) with columns such as `profileUrl`, `postContent`, and `postTimestamp`.

2. Update the configuration in `config.yaml`:
   - Set input/output file names, sheet names, and directories
   - Specify the key columns for profile matching (see `files.linkedin_posts.key_column_main_df` and `key_column_li_df`)
   - Adjust rate limits, worker count, and runtime options (test mode, dry run, error thresholds)
   - Configure persona guidance and use case definitions

3. Run the script:
```bash
python main.py
```

The script will:
- Validate and merge the input files using robust normalisation of LinkedIn profile URLs
- Only process rows with valid LinkedIn activity or referenceable data points
- Generate icebreakers and other email components using Gemini AI
- Save results to an Excel file in the `output` directory, including a `shortenedCompanyName` column
- Provide detailed logging in the log file specified in `config.yaml`

## Data Validation & Matching

- The script normalises LinkedIn profile URLs (removing protocol, www, trailing slashes, etc.) for robust matching between files
- Only rows with valid LinkedIn activity (recent posts) or valid `referenceableDataPoints` are processed
- The output file contains only processed rows, not a copy of the input
- The `shortenedCompanyName` column is generated for each row and included in the output

## Error Handling & Logging

- Comprehensive error handling for API rate limits, network issues, invalid responses, file I/O errors, and data parsing
- Errors are categorised and logged with specific error types
- Logs are written to both console and the log file specified in `config.yaml`

## Configuration

All file paths, column names, and runtime settings are controlled via `config.yaml`:
- Input/output file names, directories, and sheet names
- LinkedIn posts file and key columns for matching
- Rate limits, worker count, test mode, dry run, and error thresholds
- Persona guidance and use case definitions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 