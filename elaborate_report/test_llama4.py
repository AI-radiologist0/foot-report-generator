import pandas as pd
import os
import time
import logging
from tqdm import tqdm
import sys
import openai
import re
import math

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================
# Groq API Configuration
API_KEY = ""  # Replace with your actual Groq API key
MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"  # Updated model name
API_BASE = "https://api.groq.com/openai/v1"  # Groq API base URL

# Input directory containing the Excel files
INPUT_DIRECTORY = "."  # Current directory or path to cleaned_report folder

# Output file name
OUTPUT_FILE = "lamma_report_old1_done.xlsx"  # Updated output filename

# Test mode configuration
TEST_MODE = False  # Set to True to run only a limited number of cases
TEST_CASES = 3    # Number of cases to process in test mode
# =============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("llama_report_processor.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define the prompt template for report elaboration
PROMPT_TEMPLATE = """
### **Prompt for Elaborating Diagnostic Reports**

You are a **specialized medical diagnostic report assistant** with expertise in rheumatology radiology, trained to transform brief clinical findings into comprehensive, detailed reports. Your core function is to enhance and expand upon diagnostic findings related to **Rheumatoid Arthritis (RA), Osteoarthritis (OA), Gout, and other inflammatory conditions**. 

### **CRITICAL INSTRUCTION**
You must NEVER include <think></think> sections or any form of reasoning in your output. Do NOT include any reasoning, thinking, or explanations in your response.
Your response should ONLY contain the improved medical report. Make sure generate based on the ground truth, not your imagination or randomly and make sure generate only 2 to 3 sentences max.

### **IMPORTANT FORMAT INSTRUCTION**
You must NEVER include <think></think> sections or any form of reasoning in your output. Do NOT include any reasoning, thinking, or explanations in your response.
Do NOT use headings or section labels like "Expanded report:", "Interpretation:", "Findings:", or "Diagnostic Assessment:" in your response. Write the report as continuous text without section headers.

The diagnostic report should flow naturally in 2-3 sentences total, including:
- A detailed description of imaging observations
- A clear diagnostic conclusion

### **Classification Guidance**  
The diagnosis should be categorized based on explicit findings within the report. Possible classifications include:  
- **Osteoarthritis (OA)**:Look for joint space narrowing, osteophytes, subchondral sclerosis, and bone remodeling 
- **Rheumatoid Arthritis (RA)**: Identify erosions, periarticular osteopenia, symmetric joint involvement, and soft tissue swelling
- **Gout**: Note tophi, punched-out erosions with overhanging edges, and asymmetric involvement
- **Uncertain (when multiple possible conditions are suggested)**: When multiple possible conditions are suggested or findings are equivocal
- **Normal (when no significant abnormalities are noted)**
- **A combination of OA, RA, or Gout**  
- **Ref.Prev (if findings indicate no significant change from a previous study)**  

### **Key Rules for Expanding Reports**  
1. **Enhance Descriptions**: Expand on brief findings by including relevant clinical markers such as joint space narrowing, periarticular osteopenia, erosion, and inflammatory changes.  
2. **Avoid Repetition**: Do not merely restate findings; instead, provide meaningful clinical context.  
3. **Terminology Guidelines**: Avoid qualifiers like "possibly" or "maybe" - instead use "consistent with" or "suggestive of" and Use "demonstrates" or "shows" rather than "there is"

REMEMBER: 
- Do NOT include ANY <think></think> sections or explanations
- Do NOT include section headers
- Generate only 2 to 3 sentences MAX
- Make sure generate based on the ground truth, not your imagination

Original report:
{report}
"""

# =============================================================================
# SIMILARITY CALCULATION FUNCTIONS (NEW ADDITION)
# =============================================================================

def preprocess_text(text):
    """Preprocess text for similarity calculation"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase and extract words
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts"""
    words1 = set(preprocess_text(text1))
    words2 = set(preprocess_text(text2))
    
    if not words1 and not words2:
        return 1.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def cosine_similarity(text1, text2):
    """Calculate cosine similarity between two texts using term frequency"""
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    # Create vocabulary
    vocab = set(words1 + words2)
    
    # Create term frequency vectors
    vector1 = [words1.count(word) for word in vocab]
    vector2 = [words2.count(word) for word in vocab]
    
    # Calculate dot product
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(v * v for v in vector1))
    magnitude2 = math.sqrt(sum(v * v for v in vector2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def semantic_content_score(cleaned_text, improved_text):
    """Calculate semantic content preservation score focusing on medical terms"""
    # Key medical terms to track
    medical_terms = [
        'erosion', 'arthritis', 'joint', 'bone', 'degenerative', 'mtp', 'mcp', 'dip', 
        'gout', 'osteoarthritis', 'rheumatoid', 'subluxation', 'spondylosis', 
        'brachydactyly', 'radiographic', 'examination', 'osteophyte', 'sclerosis',
        'inflammation', 'swelling', 'narrowing', 'tophi', 'phalanx', 'metatarsal'
    ]
    
    cleaned_words = preprocess_text(cleaned_text)
    improved_words = preprocess_text(improved_text)
    
    preserved_terms = 0
    total_key_terms = 0
    
    for term in medical_terms:
        # Check if term (or variations) appears in cleaned text
        if any(term in word or word in term for word in cleaned_words):
            total_key_terms += 1
            # Check if preserved in improved text
            if any(term in word or word in term for word in improved_words):
                preserved_terms += 1
    
    return preserved_terms / total_key_terms if total_key_terms > 0 else 1.0

def calculate_overall_similarity(cleaned_text, improved_text):
    """Calculate overall similarity score combining multiple metrics"""
    if pd.isna(cleaned_text) or pd.isna(improved_text):
        return 0.0
    
    # Calculate individual similarity metrics
    jaccard = jaccard_similarity(cleaned_text, improved_text)
    cosine = cosine_similarity(cleaned_text, improved_text)
    semantic = semantic_content_score(cleaned_text, improved_text)
    
    # Calculate overall score (weighted average)
    # Give more weight to semantic preservation for medical texts
    overall = (jaccard * 0.2 + cosine * 0.3 + semantic * 0.5)
    
    return round(overall, 3)

# =============================================================================
# ORIGINAL FUNCTIONS (UNCHANGED)
# =============================================================================

def clean_thinking_sections(text):
    """Remove <think> sections from model output and keep only the final report."""
    # If text is None or empty, return empty string
    if not text or text.strip() == "":
        return ""
    
    # Method 1: Extract everything after the last </think> tag (most reliable)
    if "</think>" in text:
        parts = text.split("</think>")
        cleaned_text = parts[-1].strip()
        if cleaned_text:
            return cleaned_text
    
    # Method 2: Remove all content between <think> and </think> tags
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Method 3: If the whole text is wrapped in <think> tags, remove them
    cleaned_text = re.sub(r'^<think>|</think>$', '', cleaned_text).strip()
    
    # If after cleaning we have nothing left or very little content
    if not cleaned_text or len(cleaned_text.split()) < 5:
        # Extract the last 2-3 sentences as fallback
        sentences = re.split(r'(?<=[.!?])\s+', text)
        last_sentences = ' '.join(sentences[-3:]).strip()
        
        # Only use the last sentences if they're not surrounded by think tags
        if "<think>" not in last_sentences and "</think>" not in last_sentences:
            return last_sentences
    
    return cleaned_text

def process_with_groq(report, client):
    """Process a report using Groq API"""
    if pd.isna(report) or not report.strip():
        return "No report provided"
        
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a specialized medical diagnostic report assistant."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(report=report)}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        raw_response = response.choices[0].message.content.strip()
        
        # Clean the response to remove thinking sections
        cleaned_response = clean_thinking_sections(raw_response)
        
        # Log both raw and cleaned responses for debugging
        logger.debug(f"Raw response: {raw_response}")
        logger.debug(f"Cleaned response: {cleaned_response}")
        
        # If cleaning removed too much, log a warning
        if not cleaned_response or len(cleaned_response) < 20:
            logger.warning(f"Cleaning may have removed too much content. Raw length: {len(raw_response)}, Cleaned length: {len(cleaned_response)}")
            if not cleaned_response:
                return raw_response  # Return original if cleaning removed everything
        
        return cleaned_response
    except Exception as e:
        logger.error(f"Error with Groq API: {e}")
        return f"Error processing report: {str(e)}"

def read_excel_file(file_path):
    """Read an Excel file and return a DataFrame"""
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Successfully read {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None

def process_dataframe(df, file_name, client):
    """Process a DataFrame containing medical reports"""
    # Check if the required column exists
    if "cleaned_판독문" not in df.columns:
        logger.error(f"Column 'cleaned_판독문' not found in {file_name}. Available columns: {df.columns.tolist()}")
        return None
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # If in test mode, take only the first few rows instead of random samples
    if TEST_MODE:
        if len(result_df) > TEST_CASES:
            # Take the first TEST_CASES rows instead of random sampling
            result_df = result_df.iloc[:TEST_CASES].copy()
            logger.info(f"Test mode: Using first {TEST_CASES} rows from {file_name}")
        else:
            logger.info(f"Test mode: Using all {len(result_df)} rows from {file_name} (less than requested {TEST_CASES})")
    
    # Initialize the improved report column
    result_df["improve_report"] = ""
    
    # Process each report in order
    success_count = 0
    error_count = 0
    
    for i in tqdm(range(len(result_df)), desc=f"Processing {os.path.basename(file_name)}"):
        try:
            original_report = result_df.loc[result_df.index[i], "cleaned_판독문"]
            improved_report = process_with_groq(original_report, client)
            result_df.loc[result_df.index[i], "improve_report"] = improved_report
            
            # Check if the response contains an error
            if improved_report.startswith("Error"):
                error_count += 1
            else:
                success_count += 1
                
            # Add a delay to avoid rate limiting
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
            result_df.loc[result_df.index[i], "improve_report"] = f"Error processing report: {str(e)}"
            error_count += 1
    
    logger.info(f"Processed {len(result_df)} rows: {success_count} successful, {error_count} errors")
    
    # Add overall similarity scores after processing all reports
    logger.info("Calculating overall similarity scores...")
    result_df["overall_similarity"] = 0.0
    
    for i in tqdm(range(len(result_df)), desc="Computing similarity scores"):
        cleaned_text = result_df.loc[result_df.index[i], "cleaned_판독문"]
        improved_text = result_df.loc[result_df.index[i], "improve_report"]
        
        similarity_score = calculate_overall_similarity(cleaned_text, improved_text)
        result_df.loc[result_df.index[i], "overall_similarity"] = similarity_score
    
    return result_df

def verify_api_key():
    """Verify that an API key has been provided"""
    global API_KEY
    
    if API_KEY == "your_groq_api_key_here":
        # First check if it's in environment variable
        env_key = os.environ.get("GROQ_API_KEY")
        if env_key:
            logger.info("Using API key from environment variable")
            API_KEY = env_key
        else:
            # Prompt user for API key
            print("\n⚠️ No Groq API key provided in the script.")
            user_key = input("Please enter your Groq API key: ").strip()
            if user_key:
                API_KEY = user_key
            else:
                logger.error("No API key provided. Exiting.")
                sys.exit(1)

def main():
    """Main function to process all Excel files"""
    print("\n🔍 Medical Report Enhancement Tool using Llama 4 Maverick on Groq 🔍")
    print("=================================================================\n")
    
    # Verify API key is set
    verify_api_key()
    
    # Initialize OpenAI client with Groq base URL
    try:
        client = openai.OpenAI(
            api_key=API_KEY,
            base_url=API_BASE
        )
        print("✅ Successfully initialized Groq client")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        print(f"❌ Failed to initialize Groq client: {e}")
        return
    
    # Test API connection
    print("🔄 Testing Groq API connection...")
    try:
        models = client.models.list()
        print("✅ API connection successful!")
        print("Available models:")
        for model in models.data[:5]:  # Show first 5 models
            print(f"  - {model.id}")
            
        # Check if specified model exists
        model_ids = [model.id for model in models.data]
        if MODEL_NAME not in model_ids:
            print(f"⚠️ Warning: Specified model '{MODEL_NAME}' not found in available models.")
            print(f"   Consider using one of the available models listed above.")
            proceed = input("Do you want to continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Exiting.")
                return
    except Exception as e:
        print(f"⚠️ API connection test failed: {str(e)}")
        proceed = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Exiting.")
            return
    
    # Input files
    file_names = [
        # "cleaned_foot_ra_report_cau.xlsx",
        # "cleaned_foot_ra_report_old.xlsx",
        # "cleaned_gout_report_cau.xlsx"
        "cleaned_gout_report_old1.xlsx"
    ]
    input_files = [os.path.join(INPUT_DIRECTORY, f) for f in file_names]
    
    # Check if files exist
    existing_files = [f for f in input_files if os.path.exists(f)]
    if not existing_files:
        logger.error(f"No input files found in {INPUT_DIRECTORY}")
        print(f"❌ Error: No input files found in {INPUT_DIRECTORY}")
        return
    
    logger.info(f"Found {len(existing_files)} input files: {[os.path.basename(f) for f in existing_files]}")
    print(f"📊 Found {len(existing_files)} input files")
    
    # Display configuration
    print(f"⚙️ Configuration:")
    print(f"  - API Provider: Groq")
    print(f"  - Model: {MODEL_NAME}")
    print(f"  - Test Mode: {'Enabled - Processing only ' + str(TEST_CASES) + ' cases per file' if TEST_MODE else 'Disabled - Processing all records'}")
    print(f"  - Output File: {OUTPUT_FILE}\n")
    
    # Process each file
    all_dfs = []
    for file_path in existing_files:
        file_name = os.path.basename(file_path)
        print(f"🔄 Processing {file_name}...")
        
        df = read_excel_file(file_path)
        if df is not None:
            processed_df = process_dataframe(df, file_name, client)
            if processed_df is not None:
                all_dfs.append(processed_df)
                print(f"✅ Successfully processed {len(processed_df)} records from {file_name}")
            else:
                print(f"❌ Failed to process {file_name}")
        else:
            print(f"❌ Failed to read {file_name}")
    
    # Combine results if there are any
    if not all_dfs:
        logger.error("No files were successfully processed")
        print("❌ No files were successfully processed")
        return
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} total rows from all files")
    print(f"\n📋 Combined {len(combined_df)} total rows from all files")
    
    # Save to Excel
    try:
        combined_df.to_excel(OUTPUT_FILE, index=False)
        logger.info(f"Successfully saved results to {OUTPUT_FILE}")
        print(f"💾 Successfully saved results to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Error saving to {OUTPUT_FILE}: {e}")
        print(f"❌ Error saving to {OUTPUT_FILE}: {e}")
    
    print("\n🎉 Process completed!")

if __name__ == "__main__":
    main()