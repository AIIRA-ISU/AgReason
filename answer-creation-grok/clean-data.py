import json
import os
from tqdm import tqdm
import markdown
from xhtml2pdf import pisa

# Function to convert HTML to PDF using xhtml2pdf
def convert_html_to_pdf(source_html, output_filename):
    full_html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>Document</title>
      </head>
      <body>
        {source_html}
      </body>
    </html>
    """
    with open(output_filename, "wb") as result_file:
        pisa_status = pisa.CreatePDF(full_html, dest=result_file)
    return pisa_status.err

# Directory paths
jsonl_dir = "data/jsonl_results"

# Define the base directory for saving analysis outputs
base_dir = "data/insect_captions_data"
# Define separate base directories for each format inside the parent folder
txt_base_dir = os.path.join(base_dir, "txt")
md_base_dir = os.path.join(base_dir, "markdown")
pdf_base_dir = os.path.join(base_dir, "pdf")

# Create these base directories if they don't exist
os.makedirs(txt_base_dir, exist_ok=True)
os.makedirs(md_base_dir, exist_ok=True)
os.makedirs(pdf_base_dir, exist_ok=True)

# Loop through all .jsonl files in the jsonl_results directory
for jsonl_filename in os.listdir(jsonl_dir):
    if not jsonl_filename.endswith(".jsonl"):
        continue

    jsonl_filepath = os.path.join(jsonl_dir, jsonl_filename)
    with open(jsonl_filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Process each line in the JSONL file with a progress bar
    for idx, line in enumerate(tqdm(lines, desc=f"Processing {jsonl_filename}"), start=1):
        try:
            data = json.loads(line)
            
            # Extract the analysis text from the fixed key path
            analysis_text = data[1]["choices"][0]["message"]["content"]
            
            # Extract insect metadata from the last element in the JSON array
            metadata = data[-1]
            insect_name = metadata["insect_name"]
            file_name = metadata["file_name"]
            base_file_name = os.path.splitext(file_name)[0]  # Remove the .jpg extension
            
            # Create subfolders for the insect inside each base directory
            txt_folder = os.path.join(txt_base_dir, insect_name)
            md_folder = os.path.join(md_base_dir, insect_name)
            pdf_folder = os.path.join(pdf_base_dir, insect_name)
            os.makedirs(txt_folder, exist_ok=True)
            os.makedirs(md_folder, exist_ok=True)
            os.makedirs(pdf_folder, exist_ok=True)
            
            # Define file paths for plain text, Markdown, and PDF outputs
            output_txt_file = os.path.join(txt_folder, base_file_name + ".txt")
            output_md_file = os.path.join(md_folder, base_file_name + ".md")
            output_pdf_file = os.path.join(pdf_folder, base_file_name + ".pdf")
            
            # If all destination files already exist, skip to the next line
            if os.path.exists(output_txt_file) and os.path.exists(output_md_file) and os.path.exists(output_pdf_file):
                continue

            # Save the plain text analysis
            with open(output_txt_file, "w", encoding="utf-8") as txt_file:
                txt_file.write(analysis_text)
            
            # Prepare and save the Markdown version
            md_text = f"### Analysis for {insect_name} (File: {base_file_name})\n\n{analysis_text}\n\n---"
            with open(output_md_file, "w", encoding="utf-8") as md_file:
                md_file.write(md_text)
            
            # Convert Markdown to HTML then to PDF and save it
            html_content = markdown.markdown(md_text)
            err = convert_html_to_pdf(html_content, output_pdf_file)
            if err:
                print(f"Error converting {output_md_file} to PDF.")
                
        except Exception as e:
            print(f"Error processing {jsonl_filename} line {idx}: {e}")

print("Processing complete.")
