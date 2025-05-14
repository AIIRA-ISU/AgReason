import re
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

_ref = None  # Global reference shared by processes

def init_pool(ref_obj):
    global _ref
    _ref = ref_obj

# ========= Helper Functions ==========

def get_crops_from_sheet(data, sheet_name):
    crops_dict = {}
    df = data[sheet_name]
    for col in df.columns:
        crop_list = df[col].dropna().tolist()
        crops_dict[col] = [str(x).strip() for x in crop_list]
    return crops_dict

def get_list_from_sheet(data, sheet_name, columns):
    df = data[sheet_name]
    res = {}
    for col in columns:
        res[col] = df[col].dropna().tolist()
    return res

def union_sheet_values(data, sheet_name):
    if sheet_name not in data:
        return []
    df = data[sheet_name]
    values = []
    for col in df.columns:
        values.extend(df[col].dropna().tolist())
    return list(set(str(v).strip() for v in values))

def normalize_string(s):
    return re.sub(r'\s+', ' ', s).strip()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[.?!]+$', '', text)
    return text

def find_match(text, possible_values, long_text=True):
    text_normalized = normalize_text(text) if re.search(r"\s", text) and long_text else normalize_string(text)
    for value in possible_values:
        candidate = normalize_text(value) if re.search(r"\s", value) and long_text else normalize_string(value)
        pattern = re.escape(candidate) if ' ' in candidate else r'\b' + re.escape(candidate) + r'\b'
        if re.search(pattern, text_normalized, re.IGNORECASE):
            return candidate
    return None

def find_all_matches(text, possible_values):
    text_normalized = normalize_text(text) if re.search(r"\s", text) else normalize_string(text)
    matches = []
    for value in possible_values:
        candidate = normalize_text(value) if re.search(r"\s", value) else normalize_string(value)
        pattern = re.escape(candidate) if ' ' in candidate else r'\b' + re.escape(candidate) + r'\b'
        found = re.findall(pattern, text_normalized, flags=re.IGNORECASE)
        if found:
            matches.extend(found)
    return list(set(matches))

def classify_question_type_from_column(column):
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
    base_template = clean_for_template_search(modified_question)
    normalized_base = re.sub(r'\s+', '', base_template)
    for col, templates in templates_ref.items():
        for template in templates:
            if normalized_base == re.sub(r'\s+', '', template):
                return classify_question_type_from_column(col), col, template.strip(), base_template
    return None, None, None, base_template

def build_whitespace_tolerant_pattern(s):
    s = normalize_text(s) if re.search(r"\s", s) else normalize_string(s)
    return r'\b' + r'\s*'.join(re.escape(part) for part in s.split()) + r'\b'

def extract_metadata(question, ref):
    metadata = {}

    if re.search(ref['farming_practices']['organic'], question, re.IGNORECASE):
        metadata['farming_practice'] = 'organic'
    elif re.search(ref['farming_practices']['conventional'], question, re.IGNORECASE):
        metadata['farming_practice'] = 'conventional'

    if re.search(ref['farm_sizes']['Commercial'], question, re.IGNORECASE):
        metadata['farm_size'] = 'Commercial'
    elif re.search(ref['farm_sizes']['Small'], question, re.IGNORECASE):
        metadata['farm_size'] = 'Small'

    if 'farming_practice' in metadata and 'farm_size' in metadata:
        metadata['crop_source'] = {
            ('organic', 'Commercial'): 'Y_O_C_Variables',
            ('organic', 'Small'): 'Y_O_S_Variables',
            ('conventional', 'Commercial'): 'Y_C_C_Variables',
            ('conventional', 'Small'): 'Y_C_S_Variables'
        }.get((metadata['farming_practice'], metadata['farm_size']), 'YxL_Variables')
    else:
        metadata['crop_source'] = 'YxL_Variables'

    loc = find_match(question, ref['locations'], long_text=False)
    if loc:
        metadata['location'] = loc
        question = re.sub(r'\b' + re.escape(loc) + r'\b', 'L', question, flags=re.IGNORECASE)

    crop_source = metadata.get('crop_source', 'YxL_Variables')
    location_detected = metadata.get('location', None)

    def replace_crop(crop_list):
        crop = find_all_matches(question, crop_list)
        if crop:
            longest_crop = max(crop, key=len)
            metadata['crop'] = longest_crop
            question_mod = re.sub(build_whitespace_tolerant_pattern(longest_crop), 'Y', question, flags=re.IGNORECASE)
            return question_mod
        return question

    if location_detected and crop_source in ref['crops'] and location_detected in ref['crops'][crop_source]:
        question = replace_crop(ref['crops'][crop_source][location_detected])
    else:
        for loc_key, crop_list in ref['crops'].get(crop_source, {}).items():
            question = replace_crop(crop_list)
            if 'crop' in metadata:
                if 'location' not in metadata:
                    metadata['location'] = loc_key
                    question = re.sub(r'\b' + re.escape(loc_key) + r'\b', 'L', question, flags=re.IGNORECASE)
                break

    farm_size = metadata.get('farm_size', 'Small')
    for mod_cat, mod_values in ref['modifiers'].get(farm_size, {}).items():
        mods_found = find_match(question, mod_values)
        if mods_found:
            metadata[mod_cat] = mods_found
            #for mod in mods_found:
            question = re.sub(build_whitespace_tolerant_pattern(mods_found), mod_cat, question, flags=re.IGNORECASE)

    def find_variable(var_type):
        var_found, match_to_cat = [], {}
        for cat, vals in ref.get(var_type, {}).items():
            matches = find_all_matches(question, vals)
            var_found.extend(matches)
            for m in matches:
                match_to_cat[m] = cat
        return var_found, match_to_cat

    g_found, g_map = find_variable('G_variables')
    if g_found:
        g = max(g_found, key=len)
        metadata['G'] = g
        metadata['G_type'] = g_map[g]
        repl_g = 'C' if metadata['G_type'] == 'C_CoverCrop' else 'G'
    else:
        g, repl_g = None, None

    x_found, x_map = find_variable('X_variables')
    if x_found:
        x = max(x_found, key=len)
        metadata['X'] = x
        metadata['X_type'] = x_map[x]
        repl_x = 'X'
    else:
        x, repl_x = None, None

    if x and g:
        if len(x) >= len(g):
            question = re.sub(build_whitespace_tolerant_pattern(x), repl_x, question, flags=re.IGNORECASE)
        else:
            question = re.sub(build_whitespace_tolerant_pattern(g), repl_g, question, flags=re.IGNORECASE)
    elif x:
        question = re.sub(build_whitespace_tolerant_pattern(x), repl_x, question, flags=re.IGNORECASE)
    elif g:
        question = re.sub(build_whitespace_tolerant_pattern(g), repl_g, question, flags=re.IGNORECASE)

    if 'question_templates' in ref:
        inferred_qtype, tpl_col, matched_template, base_template = detect_question_type_from_templates(question, ref['question_templates'])
        metadata['question_type'] = inferred_qtype or 'Unknown'
        if tpl_col: metadata['template_column'] = tpl_col
        if matched_template: metadata['matched_template'] = matched_template
    else:
        base_template = clean_for_template_search(question)

    return question, base_template, metadata

def process_question(q):
    modified_q, base_template, metadata = extract_metadata(q, _ref)
    return {
        'original_question': q,
        'modified_question': modified_q,
        'base_template': base_template,
        **metadata
    }

# ========= Main Pipeline ==========

def main():
    master_path = 'data/MasterAgAIQuestionFramework.xlsx'
    data = pd.read_excel(master_path, sheet_name=None)

    ref = {
        'farming_practices': {
            'organic': r'I use organic practices.',
            'conventional': r'I use conventional practices.'
        },
        'farm_sizes': {
            'Commercial': r'I am on a commercial farm.',
            'Small': r'I am on a small farm.'
        },
        'locations': [str(loc).strip() for loc in data['L_Variables']['L_List'].dropna().tolist()],
        'crops': {
            sheet: get_crops_from_sheet(data, sheet)
            for sheet in ["YxL_Variables", "Y_O_C_Variables", "Y_O_S_Variables", "Y_C_C_Variables", "Y_C_S_Variables"]
        },
        'modifiers': {
            'Small': get_list_from_sheet(data, "SmallScale_Modifiers", ["SmallScale_Weather", "SmallScale_Random", "SmallScale_PlantingTime"]),
            'Commercial': get_list_from_sheet(data, "LargeScale_Modifiers", ["LargeScale_Random", "LargeScale_Deficiencies", "LargeScale_Field", "LargeScale_Weather", "LargeScale_PlantingTime"])
        },
        'X_variables': {
            'X_Soil': data['X_Variables']['X_Soil'].dropna().tolist(),
            'X_Weather': data['X_Variables']['X_Weather'].dropna().tolist(),
            'X_InSeason_Nutrients': data['X_Variables']['X_InSeason_Nutrients'].dropna().tolist(),
            'X_InSeason_Other': data['X_Variables']['X_Inseason_Other'].dropna().tolist()
            if 'X_Inseason_Other' in data['X_Variables'] else data['X_Variables']['X_InSeason_Other'].dropna().tolist(),
            'X_OutsideSeason': data['X_Variables']['X_OutsideSeason'].dropna().tolist(),
            'X_Management': data['X_Variables']['X_Management'].dropna().tolist(),
            'X_Disease': union_sheet_values(data, 'X_Disease_Variables'),
            'X_Insect': union_sheet_values(data, 'X_Insect_Variables'),
            'X_Weed': union_sheet_values(data, 'X_Weed_Variables')
        },
        'G_variables': {
            'G_Soil': data['G_C_Variables']['G_Soil'].dropna().tolist(),
            'G_CoverCrop': data['G_C_Variables']['G_CoverCrop'].dropna().tolist(),
            'C_CoverCrop': data['G_C_Variables']['C_CoverCrop'].dropna().tolist()
        },
        'question_templates': {
            col: data['QuestionTemplates'][col].dropna().tolist()
            for col in data['QuestionTemplates'].columns
        }
    }

    questions = pd.read_excel('new_generations/flowchart/80k.xlsx')['questions'].tolist()

    with ProcessPoolExecutor(initializer=init_pool, initargs=(ref,)) as executor:
        processed_data = list(tqdm(executor.map(process_question, questions), total=len(questions), desc="Processing in parallel"))

    df = pd.DataFrame(processed_data)
    df.drop(columns=["matched_template", "question_type", "base_template", "modified_question"], errors="ignore", inplace=True)
    #df.to_excel('new_generations/flowchart/0.5k_with_metadata.xlsx', index=False)
    df.to_json('new_generations/flowchart/80k_with_metadata.json', orient='records', indent=2, force_ascii=False)

if __name__ == "__main__":
    main()
