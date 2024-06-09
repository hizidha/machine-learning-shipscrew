import requests
import pandas as pd
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


gender_encoding = {'WANITA': 0, 'PRIA': 1}
status_encoding = {'OFF': 0, 'ON BOARD': 1}
edu_level_encoding = {'SD': 0, 'SMP': 1, 'SMA': 2, 'SLTA': 2, 'STM': 2, 'SMK': 2, 'D1': 3, 
                    'D2': 4, 'D3': 5, 'D4': 6, 'S1': 6, 'S2': 7, 'S3': 8}
experience_encoding = {'<3 BULAN': 0, '3-6 BULAN': 1, '6-9 BULAN': 2, '>9 BULAN': 3}
certificate_encoding = {'BASIC SAFETY TRAINING': 0, 'ANT-D': 1, 'ATT-D': 1, 'ANT-V': 2, 'ATT-V': 2,
                        'ANT-IV': 3, 'ATT-IV': 3, 'ANT-III': 4, 'ATT-III': 4, 'ANT-II': 5, 
                        'ATT-II': 5, 'ETR': 5, 'ANT-I': 6, 'ATT-I': 6, 'ETO': 6}

encoding_dicts = {
    'GENDER': gender_encoding,
    'STATUS': status_encoding,
    'EDU_LEVEL': edu_level_encoding,
    'EXPERIENCE': experience_encoding,
    'CERTIFICATE': certificate_encoding
}

def encode_columns(df):
    for column, encoding in encoding_dicts.items():
        df[column + '_EN'] = df[column].map(encoding)
        
    return df


def getRecommendation(dataCandidates):
    # URL berbagi file Excel di Google Drive
    # url = 'https://docs.google.com/spreadsheets/d/151Kn-JdETipRu_IsJty1-tzx_5iH8TeP/export?format=xlsx'
    # response = requests.get(url)
    
    # if response.status_code == 200:
    #     with BytesIO(response.content) as f:
    #         df = pd.read_excel(f)
    # else:
    #     print("Failed:", response.status_code)
    
    df = pd.read_excel('./data/datashipsCrew.xlsx')

    df.drop('NO', axis=1, inplace=True)
    df['NAME'] = df['NAME'].str.title()
    df['GENDER'] = df['GENDER'].str.upper()
    df['LAST_POSITION'] = df['LAST_POSITION'].str.lower()
    df['AGE_EN'] = df['AGE'].copy()
    
    df = encode_columns(df)
    
    tfidf_vectorizer = TfidfVectorizer()
    last_position_tfidf = tfidf_vectorizer.fit_transform(df['LAST_POSITION']).toarray()
    
    features = ['AGE_EN', 'GENDER_EN', 'STATUS_EN', 'EDU_LEVEL_EN', 'EXPERIENCE_EN']
    df[features] = df[features].fillna(-1).astype(int)
    all_features = pd.concat([pd.DataFrame(df[features]), pd.DataFrame(last_position_tfidf)], axis=1)

    if len(dataCandidates) > 1:
        encoded_input = {
            'STATUS_EN': min(int(status_encoding[dataCandidate['STATUS']]) for dataCandidate in dataCandidates),
            'EDU_LEVEL_EN': min(int(edu_level_encoding[dataCandidate['EDU_LEVEL']]) for dataCandidate in dataCandidates),
            'AGE_EN': sum(int(dataCandidate['AGE']) for dataCandidate in dataCandidates) / len(dataCandidates),
            'GENDER_EN': max(int(gender_encoding[dataCandidate['GENDER']]) for dataCandidate in dataCandidates),
            'CERTIFICATE_EN': min(int(certificate_encoding[dataCandidate['CERTIFICATE']]) for dataCandidate in dataCandidates),
            'EXPERIENCE_EN': min(int(experience_encoding[dataCandidate['EXPERIENCE']]) for dataCandidate in dataCandidates)
        }
        all_last_positions = ';'.join(dataCandidate['LAST_POSITION'] for dataCandidate in dataCandidates)
        all_last_positions = all_last_positions.lower()
        last_position_tfidf_input = tfidf_vectorizer.transform([all_last_positions]).toarray()
    
    else:
        dataCandidate = dataCandidates[0]
        encoded_input = {
            'STATUS_EN': status_encoding[dataCandidate['STATUS']],
            'EDU_LEVEL_EN': edu_level_encoding[dataCandidate['EDU_LEVEL']],
            'AGE_EN': int(dataCandidate['AGE']),
            'GENDER_EN': gender_encoding[dataCandidate['GENDER']],
            'CERTIFICATE_EN': certificate_encoding[dataCandidate['CERTIFICATE']],
            'EXPERIENCE_EN': experience_encoding[dataCandidate['EXPERIENCE']]
        }
        last_position_tfidf_input = tfidf_vectorizer.transform([dataCandidate['LAST_POSITION']]).toarray()

    input_features = [encoded_input[feat] for feat in features]
    input_features.extend(last_position_tfidf_input[0])

    cosine_similarities = cosine_similarity([input_features], all_features).flatten()
    recommendation_indices = [idx for idx, sim in enumerate(cosine_similarities) if sim > 0 and df.loc[idx, 'CERTIFICATE_EN'] >= encoded_input['CERTIFICATE_EN']]
    recommendation_indices.sort(key=lambda idx: cosine_similarities[idx], reverse=True)
    
    recommendations = df.iloc[recommendation_indices][['NAME', 'AGE', 'GENDER', 'STATUS', 'EDU_LEVEL', 'EXPERIENCE', 'LAST_POSITION', 'CERTIFICATE']]
    recommendations['SIMILARITY (%)'] = [round(similarity * 100, 2) for similarity in cosine_similarities[recommendation_indices]]
    
    recommendations = recommendations[recommendations['SIMILARITY (%)'] >= 70]  # Filter recommendations with similarity >= 70%
    recommendations[['GENDER', 'EXPERIENCE']] = recommendations[['GENDER', 'EXPERIENCE']].apply(lambda x: x.str.title())
    recommendations[['LAST_POSITION']] = recommendations[['LAST_POSITION']].apply(lambda x: x.str.upper())

    return recommendations