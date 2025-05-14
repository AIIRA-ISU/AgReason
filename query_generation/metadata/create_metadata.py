import re
import pandas as pd
import json
from tqdm import tqdm

# ========= Helper Functions ==========

def get_crops_from_sheet(data, sheet_name):
    """
    Reads a crop sheet from the master Excel file.
    Returns a dictionary mapping each location (DataFrame column)
    to a list of crop values.
    """
    crops_dict = {}
    df = data[sheet_name]
    for col in df.columns:
        crop_list = df[col].dropna().tolist()
        crops_dict[col] = [str(x).strip() for x in crop_list]
    return crops_dict

def get_list_from_sheet(data, sheet_name, columns):
    """
    Reads specific columns from a modifiers sheet.
    Returns a dictionary mapping each column to a list of its values.
    """
    df = data[sheet_name]
    res = {}
    for col in columns:
        res[col] = df[col].dropna().tolist()
    return res

def union_sheet_values(data, sheet_name):
    """
    Given a sheet (DataFrame) with multiple columns,
    returns a list (union) of all values across all columns.
    """
    if sheet_name not in data:
        return []
    df = data[sheet_name]
    values = []
    for col in df.columns:
        values.extend(df[col].dropna().tolist())
    values = [str(v).strip() for v in values]
    return list(set(values))

def normalize_string(s):
    """
    Normalize a string by replacing multiple spaces with one space and stripping extra spaces.
    """
    return re.sub(r'\s+', ' ', s).strip()

def normalize_text(text):
    """
    Normalize text by converting to lower case, replacing multiple whitespaces with one space,
    stripping leading/trailing spaces, and removing any trailing punctuation (. ? !).
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[.?!]+$', '', text)
    return text

def find_match(text, possible_values, long_text=True):
    """
    Normalize the input text and each candidate value,
    then return the first candidate found using whole-word matching (case-insensitive).
    """
    if re.search(r"\s", text) and long_text:
        text_normalized = normalize_text(text)
    else:
        text_normalized = normalize_string(text)


    for value in possible_values:
        if re.search(r"\s", value) and long_text:
            candidate = normalize_text(value)
            pattern = re.escape(candidate)
        else:
            candidate = normalize_string(value)
            pattern = r'\b' + re.escape(candidate) + r'\b'

        if re.search(pattern, text_normalized, re.IGNORECASE):
            return candidate
    return None

def find_all_matches(text, possible_values):
    """
    Normalize the input text and each candidate value,
    then return a list of all unique candidates that match as whole words (case-insensitive).
    """
    if re.search(r"\s", text):
        text_normalized = normalize_text(text)
    else:
        text_normalized = normalize_string(text)

    matches = []
    for value in possible_values:
        if re.search(r"\s", value):
            candidate = normalize_text(value)
            pattern = re.escape(candidate)
        else:
            candidate = normalize_string(value)
            pattern = r'\b' + re.escape(candidate) + r'\b'

        
        found = re.findall(pattern, text_normalized, flags=re.IGNORECASE)
        if found:
            matches.extend(found)
    return list(set(matches))

def classify_question_type_from_column(column):
    """
    Given a template column name, return the corresponding question type.
    Mimics the logic in the generation code.
    """
    mappings = [
        ('AbioticQuestions_Soil', 'Soil'),
        ('AbioticQuestions_Weather', 'Weather'),
        ('AbioticQuestions_HarvestQuality', 'HarvestQuality'),
        ('AbioticQuestions_InSeason_Nutrients', 'InSeason_Nutrients'),
        ('AbioticQuestions_InSeason_Other', 'InSeason_Other'),
        ('AbioticQuestions_OutsideSeason', 'OutsideSeason'),
        ('BioticQuestions_Diseases', 'Diseases'),
        ('BioticQuestions_Insects', 'Insects'),
        ('BioticQuestions_Weeds', 'Weeds'),
        ('ManagementQuestions_Management', 'Management'),
        ('CropQuestions_CoverCrop', 'CoverCrop')
    ]
    for key, value in mappings:
        if key in column:
            return value
    return 'Crop'

def clean_for_template_search(modified_question):
    """
    Remove appended text (such as location, farm context, and modifiers)
    so that only the base template remains.
    Splits the text at known markers using regex to ensure exact context.
    """
    separators = [
        r"\sI live in",
        r"\sI use organic practices\.",
        r"\sI use conventional practices\.",
        r"\sI am on a commercial farm\.",
        r"\sI am on a small farm\.",
        r"\sSmallScale_",
        r"\sLargeScale_"
    ]
    
    base = modified_question
    for sep in separators:
        match = re.search(sep, base)
        if match:
            base = base[:match.start()]
            break
    return base.strip()

def detect_question_type_from_templates(modified_question, templates_ref):
    """
    Determine which original template was used by exactly matching the normalized,
    cleaned base sentence from the generated question with each normalized template.
    Returns a tuple: (question_type, template_column, matched_template) if a match is found;
    otherwise, (None, None, None).
    """
    base_template = clean_for_template_search(modified_question)
    normalized_base = re.sub(r'\s+', '', base_template)
    
    for col, templates in templates_ref.items():
        for template in templates:
            normalized_template = re.sub(r'\s+', '', template)
            if normalized_base == normalized_template:
                return classify_question_type_from_column(col), col, template.strip(), base_template
    return None, None, None, base_template

def build_whitespace_tolerant_pattern(s):
    """
    Given a string s, build a regex pattern that ignores differences in spacing.
    This splits s into words and rejoins them with a '\s*' pattern.
    For example, "Potassium fertilizer" becomes a pattern that will match regardless
    of extra or missing spaces between the words.
    """
    if re.search(r"\s", s):
        s = normalize_text(s)
    else:
        s = normalize_string(s)

    parts = s.split()
    pattern = r'\s*'.join(re.escape(part) for part in parts)
    return r'\b' + pattern + r'\b'

def extract_metadata(question, ref):
    """
    Extract metadata from the generated question.
    Extraction order:
      1. Farming practice (organic vs. conventional)
      2. Farm size (Commercial vs. Small)
      3. Crop source (from detected practice and size)
      4. Location (from L_Variables)
      5. Crop (from the appropriate crop sheet and location)
      6. Modifiers (from the appropriate modifiers sheet; now detects all occurrences)
      7. X and G variables (using union of all known values)
      8. Finally, determine the question type by matching the cleaned,
         normalized text with the original templates.
    Each detection replaces the found text with its placeholder.
    """
    #og_question = question
    #debugging_question = ""
    #if og_question!=debugging_question:
    #    return None, None, None

    metadata = {}

    # 1. Detect farming practice.
    if re.search(ref['farming_practices']['organic'], question, re.IGNORECASE):
        metadata['farming_practice'] = 'organic'
    elif re.search(ref['farming_practices']['conventional'], question, re.IGNORECASE):
        metadata['farming_practice'] = 'conventional'

    # 2. Detect farm size.
    if re.search(ref['farm_sizes']['Commercial'], question, re.IGNORECASE):
        metadata['farm_size'] = 'Commercial'
    elif re.search(ref['farm_sizes']['Small'], question, re.IGNORECASE):
        metadata['farm_size'] = 'Small'

    # 3. Determine crop source.
    if 'farming_practice' in metadata and 'farm_size' in metadata:
        if metadata['farming_practice'] == 'organic':
            metadata['crop_source'] = 'Y_O_C_Variables' if metadata['farm_size'] == 'Commercial' else 'Y_O_S_Variables'
        else:
            metadata['crop_source'] = 'Y_C_C_Variables' if metadata['farm_size'] == 'Commercial' else 'Y_C_S_Variables'
    else:
        metadata['crop_source'] = 'YxL_Variables'

    # 4. Extract location.
    loc = find_match(question, ref['locations'], long_text=False)
    if loc:
        metadata['location'] = loc
        question = re.sub(r'\b' + re.escape(loc) + r'\b', 'L', question, flags=re.IGNORECASE)

    # 5. Extract crop.
    crop_source = metadata.get('crop_source', 'YxL_Variables')
    location_detected = metadata.get('location', None)


    if location_detected and crop_source in ref['crops'] and location_detected in ref['crops'][crop_source]:
        crop_list = ref['crops'][crop_source][location_detected]
        crop = find_all_matches(question, crop_list)
        if crop:
            metadata['crop'] = crop

            c = max(crop, key=len)
            pattern = build_whitespace_tolerant_pattern(c)
            question = re.sub(pattern, 'Y', question, flags=re.IGNORECASE)
    else:
        if crop_source in ref['crops']:
            for loc_key, crop_list in ref['crops'][crop_source].items():
                crop = find_all_matches(question, crop_list)
                if crop:
                    metadata['crop'] = crop
                    if 'location' not in metadata:
                        metadata['location'] = loc_key
                        question = re.sub(r'\b' + re.escape(loc_key) + r'\b', 'L', question, flags=re.IGNORECASE)
                    
                    c = max(crop, key=len)
                    pattern = build_whitespace_tolerant_pattern(c)
                    question = re.sub(pattern, 'Y', question, flags=re.IGNORECASE)

    # 6. Extract modifiers.
    farm_size = metadata.get('farm_size', 'Small')
    if farm_size not in ref['modifiers']:
        farm_size = 'Small'
    mod_categories = ref['modifiers'][farm_size]
    # For each modifier category, gather all matching values.
    for mod_cat, mod_values in mod_categories.items():
        mods_found = find_all_matches(question, mod_values)
        if mods_found:
            metadata[mod_cat] = mods_found  # store as a list
            # Replace each occurrence using a whitespace-tolerant pattern.
            for mod in mods_found:
                pattern = build_whitespace_tolerant_pattern(mod)
                question = re.sub(pattern, mod_cat, question, flags=re.IGNORECASE)

    # 7. Extract G variables.
    g, g_found = None, []
    match_to_cat = {}

    for cat, g_vals in ref.get('G_variables', {}).items():
        matches = find_all_matches(question, g_vals)
        g_found.extend(matches)
        for match in matches:
            match_to_cat[match] = cat

    if g_found:
        g = max(g_found, key=len)
        cat = match_to_cat[g]

        metadata['G'] = g
        metadata['G_type'] = cat

        repl_g = 'C' if cat == 'C_CoverCrop' else 'G'
        # pattern = build_whitespace_tolerant_pattern(g)
        # question = re.sub(pattern, repl_g, question, flags=re.IGNORECASE)


    # 8. Extract X variables.
    x, x_found = None, []
    match_to_cat = {}

    for cat, x_vals in ref.get('X_variables', {}).items():
        matches = find_all_matches(question, x_vals)
        x_found.extend(matches)
        for match in matches:
            match_to_cat[match] = cat

    if x_found:
        x = max(x_found, key=len)
        cat = match_to_cat[x]

        metadata['X'] = x
        metadata['X_type'] = cat

        repl_x = 'X'
        # pattern = build_whitespace_tolerant_pattern(x)
        # question = re.sub(pattern, repl_x, question, flags=re.IGNORECASE)
    
    if x and g:
        if len(x) >= len(g):
            pattern = build_whitespace_tolerant_pattern(x)
            question = re.sub(pattern, repl_x, question, flags=re.IGNORECASE)
        else:
            pattern = build_whitespace_tolerant_pattern(g)
            question = re.sub(pattern, repl_g, question, flags=re.IGNORECASE)
    elif x and not g:
        pattern = build_whitespace_tolerant_pattern(x)
        question = re.sub(pattern, repl_x, question, flags=re.IGNORECASE)
    elif g and not x:
        pattern = build_whitespace_tolerant_pattern(g)
        question = re.sub(pattern, repl_g, question, flags=re.IGNORECASE)
    else:
        pass
    
    

    ## The problem now is that some of the X are getting partially replaced by C, as C is getting replaced first
    ## Maybe after computing both of the 
    ## There is another problem: some of the question contains "if i live in ...", these are also getting cut

    # 9. Determine question type from templates.
    if 'question_templates' in ref:
        inferred_qtype, tpl_col, matched_template, base_template = detect_question_type_from_templates(question, ref['question_templates'])
        if inferred_qtype:
            metadata['question_type'] = inferred_qtype
            metadata['template_column'] = tpl_col
            metadata['matched_template'] = matched_template
        else:
            metadata['question_type'] = 'Unknown'

    return question, base_template, metadata

# ========= Main Pipeline ==========

def main():
    master_path = 'data/MasterAgAIQuestionFramework.xlsx'
    data = pd.read_excel(master_path, sheet_name=None)

    # --- Build Reference Dictionaries ---
    ref = {}

    # Farming practices patterns.
    ref['farming_practices'] = {
        'organic': r'I use organic practices.',
        'conventional': r'I use conventional practices.'
    }

    # Farm sizes patterns.
    ref['farm_sizes'] = { 
        'Commercial': r'I am on a commercial farm.',
        'Small': r'I am on a small farm.'
    }

    # Locations from L_Variables sheet, column L_List.
    locations = data['L_Variables']['L_List'].dropna().tolist()
    locations = [str(loc).strip() for loc in locations]
    ref['locations'] = locations

    # Crop references from crop sheets.
    crop_sources = {}
    for sheet_name in ["YxL_Variables", "Y_O_C_Variables", "Y_O_S_Variables", "Y_C_C_Variables", "Y_C_S_Variables"]:
        crop_sources[sheet_name] = get_crops_from_sheet(data, sheet_name)
    ref['crops'] = crop_sources

    # Modifiers references from modifiers sheets.
    ref['modifiers'] = {}
    ref['modifiers']['Small'] = get_list_from_sheet(data, "SmallScale_Modifiers", ["SmallScale_Weather", "SmallScale_Random", "SmallScale_PlantingTime"])
    ref['modifiers']['Commercial'] = get_list_from_sheet(data, "LargeScale_Modifiers", ["LargeScale_Random", "LargeScale_Deficiencies", "LargeScale_Field", "LargeScale_Weather", "LargeScale_PlantingTime"])

    # X Variables references.
    x_variables = {
        'X_Soil': data['X_Variables']['X_Soil'].dropna().tolist(),
        'X_Weather': data['X_Variables']['X_Weather'].dropna().tolist(),
        'X_InSeason_Nutrients': data['X_Variables']['X_InSeason_Nutrients'].dropna().tolist(),
        'X_InSeason_Other': data['X_Variables']['X_Inseason_Other'].dropna().tolist() if 'X_Inseason_Other' in data['X_Variables'].columns else data['X_Variables']['X_InSeason_Other'].dropna().tolist(),
        'X_OutsideSeason': data['X_Variables']['X_OutsideSeason'].dropna().tolist(),
        'X_Management': data['X_Variables']['X_Management'].dropna().tolist(),
    }
    # Add values from extra sources.
    x_variables['X_Disease'] = union_sheet_values(data, 'X_Disease_Variables')
    x_variables['X_Insect'] = union_sheet_values(data, 'X_Insect_Variables')
    x_variables['X_Weed'] = union_sheet_values(data, 'X_Weed_Variables')
    ref['X_variables'] = x_variables

    # G Variables references. 
    ref['G_variables'] = {
        'G_Soil': data['G_C_Variables']['G_Soil'].dropna().tolist(),
        'G_CoverCrop': data['G_C_Variables']['G_CoverCrop'].dropna().tolist(),
        'C_CoverCrop': data['G_C_Variables']['C_CoverCrop'].dropna().tolist()
    }

    # Build question templates reference from the QuestionTemplates sheet.
    templates_sheet = data['QuestionTemplates']
    templates_ref = {}
    for col in templates_sheet.columns:
        templates_ref[col] = templates_sheet[col].dropna().tolist()
    ref['question_templates'] = templates_ref

    # --- Load Generated Questions ---
    questions_df = pd.read_excel('new_generations/flowchart/0.5k.xlsx')
    questions = questions_df['questions'].tolist()

    # --- Process Each Question to Extract Metadata ---
    processed_data = []
    for q in tqdm(questions, desc="Processing Questions"):
        modified_q, base_template, metadata = extract_metadata(q, ref)
        processed_data.append({
            'original_question': q,
            'modified_question': modified_q,
            'base_template': base_template,
            'metadata': metadata
        })

    # Flatten metadata dict for DataFrame conversion
    flat_data = []
    for item in processed_data:
        flat_item = {
            'original_question': item['original_question'],
            'modified_question': item['modified_question'],
            'base_template': item['base_template'],
        }
        if isinstance(item['metadata'], dict):
            flat_item.update(item['metadata'])  # Add metadata fields as columns
        flat_data.append(flat_item)

    output_excel_path = 'new_generations/flowchart/0.5k_with_metadata.xlsx'
    pd.DataFrame(flat_data).to_excel(output_excel_path, index=False)
    print("Metadata extraction saved to Excel at", output_excel_path)

    # Save output as JSON.
    #output_path = 'new_generations/flowchart/0.5k_with_metadata.json'
    #with open(output_path, "w", encoding="utf-8") as f:
    #    json.dump(processed_data, f, indent=2)
    #print("Metadata extraction (with question type) completed and saved to", output_path)

if __name__ == "__main__":
    main()
