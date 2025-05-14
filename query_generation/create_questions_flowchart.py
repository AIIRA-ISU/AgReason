# create_questions_flowchart.py

import random
import pandas as pd
import re
from collections import defaultdict

class AgQuestionGenerator:
    def __init__(self, excel_path, p=0.5, q=0.5, r=0.5):
        self.p = p
        self.q = q
        self.r = r
        self.data = pd.read_excel(excel_path, sheet_name=None)

    def select_question_template(self, col):
        df = self.data['QuestionTemplates']
        if col is None or col not in df.columns:
            col = random.choice(df.columns.tolist())
        question = df[col].dropna().sample(1).values[0]
        return question, col

    def get_location(self):
        return random.choice(self.data['L_Variables']['L_List'].dropna().tolist())

    def inject_location(self, question, location):
        if re.search(r'\bL\b', question):
            return re.sub(r'\bL\b', location, question)
        else:
            return question.strip() + f" I live in {location}."

    def select_crop_direct(self, location):
        crops = self.data['YxL_Variables'][location].dropna().tolist()
        return random.choice(crops) if crops else None

    def get_farm_context(self):
        is_organic = random.random() < self.q
        farm_size = 'Commercial' if random.random() < self.r else 'Small'
        return is_organic, farm_size

    def select_crop_by_practice(self, location, is_organic, farm_size):
        sheet_map = {
            ('Organic', 'Commercial'): 'Y_O_C_Variables',
            ('Organic', 'Small'): 'Y_O_S_Variables',
            ('Conventional', 'Commercial'): 'Y_C_C_Variables',
            ('Conventional', 'Small'): 'Y_C_S_Variables',
        }
        sheet = sheet_map[('Organic' if is_organic else 'Conventional', farm_size)]
        crops = self.data[sheet][location].dropna().tolist()
        return random.choice(crops) if crops else None

    def is_annual_crop(self, location, crop):
        return crop in self.data['Y_AnnualVariables'][location].dropna().tolist()

    def get_modifiers(self, farm_size, is_irrelevant_type):
        if farm_size == 'Small':
            df = self.data['SmallScale_Modifiers']
            modifiers = [
                random.choice(df['SmallScale_Weather'].dropna().tolist()),
                random.choice(df['SmallScale_Random'].dropna().tolist())
            ]
            if is_irrelevant_type:
                modifiers.append(random.choice(df['SmallScale_PlantingTime'].dropna().tolist()))
        else:
            df = self.data['LargeScale_Modifiers']
            modifiers = [
                random.choice(df['LargeScale_Random'].dropna().tolist()),
                random.choice(df['LargeScale_Deficiencies'].dropna().tolist()),
                random.choice(df['LargeScale_Field'].dropna().tolist()),
                random.choice(df['LargeScale_Weather'].dropna().tolist())
            ]
            if is_irrelevant_type:
                modifiers.append(random.choice(df['LargeScale_PlantingTime'].dropna().tolist()))
        return ' '.join(modifiers)

    def classify_question_type(self, column):
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

    def replace_placeholders(self, question, qtype, location, crop):
        if crop is None:
            return None
        d = self.data

        if crop:
            question = re.sub(r'\bY\b', crop, question)

        if re.search(r'\bX\b', question):
            if qtype == 'Soil':
                x = random.choice(d['X_Variables']['X_Soil'].dropna().tolist())
                g = random.choice(d['G_C_Variables']['G_Soil'].dropna().tolist())
                question = re.sub(r'\bX\b', x, question)
                question = re.sub(r'\bG\b', g, question)
            elif qtype == 'Weather':
                x = random.choice(d['X_Variables']['X_Weather'].dropna().tolist())
                question = re.sub(r'\bX\b', x, question)
            elif qtype == 'InSeason_Nutrients':
                x = random.choice(d['X_Variables']['X_InSeason_Nutrients'].dropna().tolist())
                question = re.sub(r'\bX\b', x, question)
            elif qtype == 'InSeason_Other':
                x = random.choice(d['X_Variables']['X_InSeason_Other'].dropna().tolist())
                question = re.sub(r'\bX\b', x, question)
            elif qtype == 'OutsideSeason':
                x = random.choice(d['X_Variables']['X_OutsideSeason'].dropna().tolist())
                question = re.sub(r'\bX\b', x, question)
            elif qtype == 'Diseases':
                col = f"{location}: {crop}"
                if col in d['X_Disease_Variables'].columns:
                    x = random.choice(d['X_Disease_Variables'][col].dropna().tolist())
                    question = re.sub(r'\bX\b', x, question)
                else:
                    return None  # discard question if disease variable not found
            elif qtype == 'Insects':
                if location in d['X_Insect_Variables'].columns:
                    options = d['X_Insect_Variables'][location].dropna().tolist()
                    if not options:
                        return None
                    x = random.choice(options)
                else:
                    return None
                question = re.sub(r'\bX\b', x, question)
            elif qtype == 'Weeds':
                if location in d['X_Weed_Variables'].columns:
                    options = d['X_Weed_Variables'][location].dropna().tolist()
                    if not options:
                        return None
                    x = random.choice(options)
                else:
                    return None
                question = re.sub(r'\bX\b', x, question)
            elif qtype == 'Management':
                x = random.choice(d['X_Variables']['X_Management'].dropna().tolist())
                question = re.sub(r'\bX\b', x, question)

        if qtype == 'CoverCrop':
            g = random.choice(d['G_C_Variables']['G_CoverCrop'].dropna().tolist())
            c = random.choice(d['G_C_Variables']['C_CoverCrop'].dropna().tolist())
            question = re.sub(r'\bG\b', g, question)
            question = re.sub(r'\bC\b', c, question)

        return question

    def generate_question(self, col=None):
        question, col = self.select_question_template(col)

        if not re.search(r'\bY\b', question):
            return None
        
        qtype = self.classify_question_type(col)
        location = self.get_location()
        question = self.inject_location(question, location)

        self.qtype = qtype # usef while computing statistics in generate_batch

        is_organic, farm_size = None, None
        if random.random() < self.p:
            crop = self.select_crop_direct(location)
        else:
            is_organic, farm_size = self.get_farm_context()
            question += f" I use {'organic' if is_organic else 'conventional'} practices."
            question += f" I am on a {'commercial' if farm_size == 'Commercial' else 'small'} farm."
            crop = self.select_crop_by_practice(location, is_organic, farm_size)

        needs_annual = qtype in [
            'InSeason_Nutrients', 'Soil', 'Weather', 'HarvestQuality',
            'Diseases', 'Insects', 'Weeds', 'Management', 'Crop']
        is_irrelevant = qtype in ['CoverCrop', 'InSeason_Other', 'OutsideSeason']

        if needs_annual and (not crop or not self.is_annual_crop(location, crop)):
            return self.generate_question()

        if farm_size:
            question += ' ' + self.get_modifiers(farm_size, is_irrelevant)


        return self.replace_placeholders(question, qtype, location, crop)
    
    def generate_batch(self, n, col=None, max_attempts_per_question=10, verbose=True):
        questions = set()
        qtype_counts = defaultdict(int)
        attempts = 0

        while len(questions) < n and attempts < n * max_attempts_per_question:
            question = self.generate_question(col=col)
            if question and question not in questions:
                questions.add(question)
                qtype_counts[self.qtype] += 1

            attempts += 1

        if verbose:
            print(f"\nGenerated {len(questions)} unique questions (attempts: {attempts})")
            print("Question type distribution:")
            for qtype, count in sorted(qtype_counts.items(), key=lambda x: -x[1]):
                print(f"  {qtype:20}: {count}")

        return list(questions)



if __name__ == "__main__":
    p, q, r = 0.8, 0.2, 0.75

    generator = AgQuestionGenerator('data/MasterAgAIQuestionFramework.xlsx')
    question = generator.generate_batch(n=50000)

    df = pd.DataFrame({'questions': question})
    df.to_excel('new_generations/flowchart/50k.xlsx', index=False)