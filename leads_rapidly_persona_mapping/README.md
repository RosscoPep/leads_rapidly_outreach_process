# Persona Mapping Module

This module provides functionality for mapping and categorising professional profiles into specific personas based on their roles, industries, and other characteristics.

## Overview

The Persona Mapping module processes structured data from OpenAI batch responses to:
- Categorise professionals into specific persona types
- Identify key characteristics and attributes
- Generate persona-specific insights
- Map personas to appropriate outreach strategies
- Incorporate activity data for enhanced persona categorisation

## Features

- Automated persona classification
- Multi-dimensional persona mapping
- Industry-specific persona templates
- Seniority-based persona adjustments
- Company stage considerations
- Custom persona definitions
- Activity data integration
- Resilient data merging

## Data Processing

The module processes two main data sources:
1. **Profile Data** (Primary Source)
   - Contains verified job titles, seniority levels, company information
   - Used as the primary source for persona categorisation
   - All profiles are preserved in the final output

2. **Activity Data** (Secondary Source)
   - Contains recent posts and activity information
   - Merged with profile data where matches are found
   - Used to enhance persona categorisation
   - Non-matching activity data is excluded to maintain data quality

## Configuration

The module uses a YAML configuration file (`config.yaml`) to define:
- Persona categories and their characteristics
- Industry mappings
- Seniority levels
- Company stages
- Custom rules and exceptions
- File paths for input and output
- OpenAI API settings

## Usage

1. Prepare your data files:
   - Profile data in `final_results.xlsx`
   - Activity data in `activity_data.xlsx` (optional)

2. Configure your settings in `config.yaml`:
   ```yaml
   paths:
     input:
       excel: "output/final_results.xlsx"
       posts_excel: "output/activity_data.xlsx"
       sheet_name: "RawDataResults"
     output:
       excel: "output/categorized_personas.xlsx"
       log: "persona_categorization.log"
   ```

3. Run the persona mapping script:
   ```bash
   python w_activity_spreadsheet_batch_to_openai_responses_personas.py
   ```

## Output

The module generates:
- Persona classifications for each profile
- Confidence scores for classifications
- Supporting evidence for classifications
- Recommended outreach strategies
- Customised messaging templates
- Activity data integration where available

## Data Resilience

The script includes several resilience features:
- Prioritises profile data over activity data
- Handles missing or mismatched data gracefully
- Provides detailed logging of data matching
- Continues processing even if activity data is unavailable
- Maintains data quality by excluding non-matching activity data

## Integration

This module works seamlessly with:
- OpenAI Batch Processing
- Railway Cloud Storage
- Excel data processing
- Custom outreach systems
- Activity data sources

## License

MIT License 