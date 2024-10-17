You are a clinical expert. With following extra knowledge in mind:

**Summary of Lesions: Enlarged Cardiac Silhouette, Pulmonary Consolidation, Atelectasis, and Pleural Abnormality**

**Enlarged Cardiac Silhouette:**

*   Symptoms: shortness of breath, fatigue, swelling in the legs and feet, chest pain, palpitations, and coughing up blood
*   Causes: pericardial effusion, cardiomegaly, increased cardiothoracic ratio, anterior mediastinal mass, and prominent epicardial fat pad
*   Clinical signs: pericardial effusion, cardiomegaly, increased cardiothoracic ratio, anterior mediastinal mass, and prominent epicardial fat pad
*   Laboratory data: pericardial fluid analysis, radiographic images, echocardiographic images, and biopsy results
*   Personal relevant history: history of pericardial effusion and cardiomegaly, smoking history, and family history of heart disease

**Pulmonary Consolidation:**

*   Symptoms: breath sounds and crackles, chest pain, fever, weakness, shortness of breath, and coughing up mucus or blood
*   Causes: blood from the bronchial tree and hemorrhage from the pulmonary artery, pulmonary edema, pus, and pulmonary infiltrate
*   Clinical signs: breath sounds and crackles, blood from the bronchial tree and hemorrhage from the pulmonary artery, pulmonary edema, pus, and pulmonary infiltrate
*   Laboratory data: elevated white blood cell count, elevated C-reactive protein (CRP) levels, and elevated procalcitonin levels
*   Personal relevant history: smoking history, chronic obstructive pulmonary disease (COPD), and family history of respiratory conditions

**Atelectasis:**

*   Symptoms: shortness of breath, coughing, chest pain, fatigue, and weight loss
*   Causes: pneumothorax, pleural effusion, lung cancer, and surfactant deficiency
*   Clinical signs: lung collapse, increased cardiothoracic ratio, pleural effusion, and pneumothorax
*   Laboratory data: chest radiography, computed tomography (CT) scans, and pulmonary function tests
*   Personal relevant history: smoking history, age, previous surgical history, and respiratory conditions

**Pleural Abnormality:**

*   Symptoms: chest pain, shortness of breath, coughing, fatigue, and weight loss
*   Causes: pleural effusion, pneumoperitoneum, atelectasis, tension pneumothorax, and symptomatic pneumothorax
*   Clinical signs: pleural effusion, atelectasis, tension pneumothorax, and pneumoperitoneum
*   Laboratory data: pleural fluid analysis, blood tests, chest radiography, CT scans, and pulmonary function tests
*   Personal relevant history: smoking history, lung cancer or other malignancies, medical history, family history, and occupational and environmental history

# Task you're going to perform

You will be given a report regarding a patient, and a json object with for you to sepculate the values of each attribute. 
You will return the speculated values according to the data type specified in the give JSON Object. You need to speculate the value, and null value is not acceptable.
Please keep your responses concise, with a maximum length of 5000 tokens.
Don't be unit (e.g., cm, L) in the value. Only numerical characters for numerical data types, and only true or false for boolean data types.

Following is an example for you:

## Report
INDICATION:  Central venous line placement.
TECHNIQUE:  Frontal chest radiograph.
COMPARISON:  Chest radiograph 12:42 today.
FINDINGS: 
A right subclavian catheter has been placed in the interim. The catheterterminates at the confluence of the brachiocephalic vein and superior venacava and if indicated could be advanced 3.7 cm for termination within thelow SVC.
There is no pleural effusion or pneumothorax. The cardiac silhouette remainsmildly enlarged. There is no focal airspace consolidation worrisome forpneumonia.
High density material is again seen in the paritally imaged colon in the leftabdomen. Cholecystectomy clips are noted. There are carotid calcificationsleft greater than right.
DIAGNOSED LESIONS: Enlarged cardiac silhouette.
AGE: 69.
GENDER: Female.

## Data Type 
{"Chest pain": "boolean", "Weight loss": "boolean", "History of COPD": "boolean", "Heartrate (beats per minute)": "numerical", "Oxygen levels (mmHg)": "numerical"}

## Healthy Patient Feature Range
{"Heartrate (beats per minute)": "60-100", "Oxygen levels (mmHg)": "75-100"}

## Speculated Values
{"Chest pain": true, "Weight loss": false, "History of COPD": true, "Heartrate (beats per minute)": 90, "Oxygen levels (mmHg)": 99.0}


Based on the following "Report", determine the potential values for the keywords listed in the 'Data Type' JSON object, which specifies the required keywords and their data types.

## IMPORTANT
- Return the results in JSON format, adhering to the data types defined in the 'Data Type' JSON object.
- Don't miss any keyword in the **Data Type** when returning speculated values.
- Return actual values instead of data types.
- Please keep your responses concise, with a maximum length of 5000 tokens.
- If uncertain, speculate the value rather than stating that you cannot answer rather than stating 'unknown' or 'not mentioned,' 
- Avoid generating repetitive results.
- Avoid returning repetitive keywords.
- Ensure the output strictly adheres to the specified JSON format outlined in the system prompt. The JSON object for keywords must be included as a string (enclosed in double quotes) under the 'description' field within the 'points' object.
- For numerical keywords, refer to the 'Healthy Patient Feature Range' JSON object for guidance. However, the values do not need to fall strictly within these ranges, as not all patients are healthy—just ensure the values are realistic.

## Report
INDICATION:  Hemodialysis line which was pulled.
COMPARISON:  .
TECHNIQUE:  Upright AP and lateral views of the chest.
FINDINGS:  The cardiac silhouette remains mildly enlarged.  The mediastinaland hilar contours are within normal limits.  Previously noted opacity withinthe right upper lobe has somewhat improved with residual linear opacitieslikely reflecting subsegmental atelectasis.  Additionally aeration of theleft lung base is improved and subsegmental atelectasis in the left lower lobeis noted.  No pleural effusion or pneumothorax is identified.  Inferior venacava filter is partially imaged.
IMPRESSION:  Interval improvement in aeration of the right upper lobe and leftlung base.  Minimal residual subsegmental atelectasis is noted in theseregions.  No hemodialysis catheter is identified.
DIAGNOSED LESIONS: No lesion found.
AGE: 42.
GENDER: F.
BODY TEMPERATURE FAHRENHEIT: 98.8.
HEART RATE PER MINUTE: 120.0.
RESPIRATORY RATE PER MINUTE: 22.0.
OXYGEN SATURATION PERCENTAGE: 100.0.
SYSTOLIC BLOOD PRESSURE mmHg: 161.0.
SYSTOLIC BLOOD PRESSURE mmHg: 110.0.
Pain 010: 110.0.
Acuity 1highest priority  5lowest priority: 3.0.


## Data Type 
{"Shortness of breath": "boolean", "Fatigue": "boolean", "Swelling in the legs and feet": "boolean", "Chest pain": "boolean", "Palpitations": "boolean", "Coughing up blood": "boolean", "Fluid buildup in the lungs": "boolean", "Confusion": "boolean", "Loss of appetite": "boolean", "Pericardial effusion": "boolean", "Cardiomegaly": "boolean", "Increased cardiothoracic ratio": "boolean", "Anterior mediastinal mass": "boolean", "Prominent epicardial fat pad": "boolean", "Elevated pericardial fluid levels (mL)": "numerical", "Increased pericardial fluid protein levels (mg/dL)": "numerical", "Presence of pericardial fluid cells": "boolean", "Radiographic images showing an increased cardiothoracic ratio": "boolean", "Measurements of cardiac diameter and thoracic diameter (cm)": "numerical", "Echocardiographic images showing an enlarged heart": "boolean", "Measurements of cardiac dimensions (cm)": "numerical", "Imaging studies showing a mass in the anterior mediastinum": "boolean", "Biopsy results showing the type of tumor or mass": "boolean", "History of pericardial effusion and cardiomegaly": "boolean", "History of anterior mediastinal mass and prominent epicardial fat pad": "boolean", "History of fluid accumulation": "boolean", "History of air accumulation": "boolean", "History of interstitial lung disease": "boolean", "Smoking History": "boolean", "COPD": "boolean", "Chronic Obstructive Pulmonary Disease": "boolean", "Medical History": "boolean", "Cardiovascular Conditions": "boolean", "Family History": "boolean", "Blood from Bronchial Tree": "boolean", "Hemorrhage from Pulmonary Artery": "boolean", "Pulmonary Edema": "boolean", "Pus": "boolean", "Pulmonary Infiltrate": "boolean", "Infection": "boolean", "Inflammation": "boolean", "Underlying Medical Conditions": "boolean", "White Blood Cell Count (cells/\u03bcL)": "numerical", "C-Reactive Protein (CRP) levels (mg/L)": "numerical", "Procalcitonin levels (ng/mL)": "numerical", "Lung Volume (L)": "numerical", "Diffusing Capacity (mL/min/mmHg)": "numerical", "Gas Exchange (mL/min/mmHg)": "numerical", "Arterial Blood Gas (ABG) analysis": "boolean", "Complete Blood Count (CBC)": "boolean", "Electrolyte Panel": "boolean", "Breath Sounds and Crackles": "boolean", "Cough": "boolean", "Fever": "boolean", "Difficulty breathing": "boolean", "Wheezing": "boolean", "Coughing up mucus": "boolean", "Difficulty swallowing": "boolean", "Age (years)": "numerical", "Oxygen levels (mmHg)": "numerical", "Body Temperature (\u00b0C)": "numerical", "Respiratory Symptoms": "boolean", "Coughing": "boolean", "Pleuritic chest pain": "boolean", "Respiratory failure": "boolean", "Breathing Rate (breaths/min)": "numerical", "Previous Surgical History": "boolean", "Respiratory Conditions": "boolean", "Infant Respiratory Distress Syndrome": "boolean", "Pneumothorax": "boolean", "Pleural Effusion": "boolean", "Lung Cancer": "boolean", "Pulmonary Tuberculosis": "boolean", "Postsurgical Atelectasis": "boolean", "Surfactant Deficiency": "boolean", "Cardiothoracic ratio": "numerical", "Lung expansion": "numerical", "Smoking history": "boolean", "Other Malignancies": "boolean", "Previous surgeries": "boolean", "Infections": "boolean", "Trauma": "boolean", "Rheumatoid arthritis": "boolean", "Asbestos exposure": "boolean", "Occupational exposure": "boolean", "Environmental exposure": "boolean", "Chronic respiratory conditions": "boolean", "Asthma": "boolean", "Pleural effusion": "boolean", "Pneumoperitoneum": "boolean", "Atelectasis": "boolean", "Symptomatic pneumothorax": "boolean", "Tension pneumothorax": "boolean", "Weight loss": "boolean", "Decreased lung expansion": "boolean", "Pleural friction rub": "boolean", "Pleural fluid analysis": "boolean", "Blood tests": "boolean", "Chest radiography": "boolean", "Computed tomography (CT) scans": "boolean", "Pulmonary function tests": "boolean", "Heart failure": "boolean", "Perforation of the intestine": "boolean", "Surgical procedure": "boolean", "Pleural effusion risk factors (1-5)": "numerical", "Pleural disease risk factors (1-10)": "numerical"}

## Healthy Patient Feature Range
{"Elevated pericardial fluid levels (mL)": "0-200", "Increased pericardial fluid protein levels (mg/dL)": "0-5", "Measurements of cardiac diameter and thoracic diameter (cm)": "15-25", "Measurements of cardiac dimensions (cm)": "10-20", "White Blood Cell Count (cells/\u03bcL)": "4,000-11,000", "C-Reactive Protein (CRP) levels (mg/L)": "0-10", "Procalcitonin levels (ng/mL)": "0-0.5", "Lung Volume (L)": "3-6", "Diffusing Capacity (mL/min/mmHg)": "20-40", "Gas Exchange (mL/min/mmHg)": "10-20", "Age (years)": "18-65", "Oxygen levels (mmHg)": "75-100", "Body Temperature (\u00b0C)": "36-37", "Breathing Rate (breaths/min)": "12-20", "Cardiothoracic ratio": "0.45-0.55", "Lung expansion": "0-10", "Pleural effusion risk factors (1-5)": "1-5", "Pleural disease risk factors (1-10)": "1-10"}

## Speculated Values