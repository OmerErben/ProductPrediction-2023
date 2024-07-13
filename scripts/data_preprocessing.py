import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re

def load_and_preprocess_data():
    nutrients_df = pd.read_csv('data/nutrients.csv')
    food_nutrients_df = pd.read_csv('data/food_nutrients.csv')
    food_train_df = pd.read_csv('data/food_train.csv')
    food_test_df = pd.read_csv('data/food_test.csv')

    food_nutrients_df = food_nutrients_df[food_nutrients_df['amount'] > 0]
    nutrients_join_df = pd.merge(food_nutrients_df, nutrients_df, how='left')

    label_encoder = LabelEncoder()
    food_train_df.category = label_encoder.fit_transform(food_train_df.category)

    nutrients_with_labels = pd.merge(left=nutrients_join_df, right=food_train_df[['idx', 'category']], on='idx')

    pivoted_df = nutrients_with_labels.pivot(index=['idx', 'category'], columns=['nutrient_id'], values=['amount']).fillna(-1)
    pivoted_df.columns = [col[1] for col in pivoted_df.columns]
    pivoted_df = pivoted_df.reset_index()
    pivoted_df['amount_of_nutrients'] = (pivoted_df.drop(['category'], axis=1) != -1).astype(int).sum(axis=1)

    return pivoted_df, food_train_df

def handle_text_data(food_train_df):
    def extract_first_non_number_word(s):
        match = re.search(r'\d+(\.\d+)?\s+(\D+)\b', str(s))
        return match.group(2) if match else None

    def extract_first_float(s):
        pattern = r"\d+\.\d+"
        match = re.search(pattern, str(s))
        if match:
            return float(match.group(0))
        else:
            pattern = r"\d+"
            match = re.search(pattern, str(s))
            return float(match.group(0)) if match else None

    food_train_df['household_serving_amounts'] = food_train_df['household_serving_fulltext'].apply(lambda unit: extract_first_float(unit))
    food_train_df['household_serving_words'] = food_train_df['household_serving_fulltext'].apply(lambda unit: extract_first_non_number_word(unit))
    food_train_df.drop('household_serving_fulltext', axis=1, inplace=True)

    return food_train_df
