"""Generate the synthetic demo dataset.

Everything in ``data/raw`` is fabricated for this assessment. No real patient
data is used. Re-run with::

    python scripts/generate_demo_data.py

The dataset deliberately includes missing fields, mixed date formats, duplicate
records, conflicting identifiers, OCR noise, malformed lab values and duplicate
documents so the cleaning pipeline has something meaningful to do.
"""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Rich, hand-written core patients
# ---------------------------------------------------------------------------

ELEANOR_DISCHARGE = (
    "Admission Date: 02/14/2026. Discharge Date: 02/19/2026. "
    "Chief Complaint: worsening shortness of breath and 6 lb weight gain over one week. "
    "History: 78-year-old woman with chronic heart failure with reduced ejection fraction "
    "(HFrEF, EF 30%), permanent atrial fibrillation on warfarin, hypertension, and stage 3 "
    "chronic kidney disease. She presented with dyspnea at rest, orthopnea, and bilateral "
    "lower-extremity edema. Examination: JVP elevated at 14 cm, bibasilar crackles, 2+ pitting "
    "edema. Labs: BNP 1450 pg/mL, creatinine 2.4 mg/dL (baseline 1.6), potassium 6.8 mmol/L, "
    "INR 6.2. Chest radiograph showed pulmonary vascular congestion and small bilateral pleural "
    "effusions. Echocardiogram: EF 30%, moderate mitral regurgitation. Hospital course: treated "
    "with IV furosemide with net negative 4.2 L, potassium lowered with sodium polystyrene "
    "sulfonate and insulin/dextrose, warfarin held for supratherapeutic INR. Diagnoses: acute "
    "decompensated heart failure, hyperkalemia, coagulopathy due to warfarin, CKD stage 3. "
    "Disposition: home with cardiology follow-up in 7 days. Discharge medications: furosemide "
    "40 mg daily, carvedilol 6.25 mg BID, sacubitril/valsartan 24/26 mg BID, spironolactone held. "
    "Follow-up labs in 3 days."
)

JAMES_CT = (
    "CT chest with intravenous contrast. Dense consolidation in the right lower lobe with air "
    "bronchograms. Small right parapneumonic pleural effusion. No pulmonary embolism. No "
    "pneumothorax. Mediastinal lymphadenopathy reactive in appearance. Impression: right lower "
    "lobe pneumonia with small parapneumonic effusion."
)

CORE_PATIENTS = [
    {
        "mrn": "MRN-100234",
        "first_name": "Eleanor",
        "last_name": "Vance",
        "dob": "02/11/1948",
        "gender": "F",
        "phone": "+1 212-555-0142",
        "email": "eleanor.vance@example.org",
        "address": {"line": "410 W 58th St", "city": "New York", "state": "NY", "postal_code": "10019"},
        "preferred_language": "English",
        "encounters": [
            {
                "class": "IMP",
                "reason": "Acute decompensated heart failure",
                "start": "2026-02-14",
                "end": "2026-02-19",
            }
        ],
        "documents": [
            {
                "type": "Discharge Summary",
                "title": "Discharge Summary - Acute Decompensated Heart Failure",
                "date": "2026-02-19",
                "author": "Dr. Anita Patel",
                "text": ELEANOR_DISCHARGE,
            },
            {
                "type": "Imaging Report",
                "title": "Echocardiogram - Transthoracic",
                "date": "2026-02-15",
                "author": "Dr. M. Rosen",
                "text": (
                    "Transthoracic echocardiogram. Left ventricle is moderately dilated with severe "
                    "global hypokinesis. Ejection fraction estimated at 30%. Moderate mitral "
                    "regurgitation. Left atrium moderately dilated. Right ventricular systolic "
                    "pressure estimated at 48 mmHg. Small pericardial effusion. No intracardiac "
                    "thrombus. Impression: HFrEF with EF 30%, moderate MR, pulmonary hypertension."
                ),
            },
            {
                "type": "Progress Note",
                "title": "Progress Note - Cardiology",
                "date": "2026-02-17",
                "author": "Dr. Anita Patel",
                "text": (
                    "Subjective: dyspnea improved, now comfortable at rest. Objective: BP 104/62, "
                    "HR 78 irregular, SpO2 96% on room air, weight down 2.8 kg. Exam: crackles at "
                    "bases, JVP 9 cm, trace edema. Labs: potassium 5.4, creatinine 2.1, INR 2.8. "
                    "Assessment: improving decompensated heart failure; hyperkalemia resolving; INR "
                    "therapeutic after warfarin hold. Plan: continue IV diuresis, transition to oral "
                    "furosemide, repeat BMP in the morning, resume warfarin at reduced dose."
                ),
            },
        ],
        "labs": [
            {"test_name": "BNP", "value": 1450, "unit": "pg/mL", "reference_range": "<100", "date": "2026-02-14"},
            {"test_name": "Creatinine", "value": 2.4, "unit": "mg/dL", "reference_range": "0.6-1.3", "date": "2026-02-14"},
            {"test_name": "Potassium", "value": 6.8, "unit": "mmol/L", "reference_range": "3.5-5.1", "date": "2026-02-14"},
            {"test_name": "INR", "value": 6.2, "unit": "ratio", "reference_range": "0.8-1.2", "date": "2026-02-14"},
            {"test_name": "Sodium", "value": 134, "unit": "mmol/L", "reference_range": "135-145", "date": "2026-02-14"},
            {"test_name": "Creatinine", "value": 2.1, "unit": "mg/dL", "reference_range": "0.6-1.3", "date": "2026-02-17"},
            {"test_name": "Potassium", "value": 5.4, "unit": "mmol/L", "reference_range": "3.5-5.1", "date": "2026-02-17"},
            {"test_name": "INR", "value": 2.8, "unit": "ratio", "reference_range": "0.8-1.2", "date": "2026-02-17"},
        ],
    },
    {
        "mrn": "100455",
        "first_name": "Marcus",
        "last_name": "Bell",
        "dob": "1962-07-03",
        "gender": "M",
        "phone": "646-555-0198",
        "address": {"line": "88 Hanson Pl", "city": "Brooklyn", "state": "NY", "postal_code": "11217"},
        "encounters": [{"class": "AMB", "reason": "Diabetes and CKD follow-up", "start": "2026-03-05"}],
        "documents": [
            {
                "type": "Progress Note",
                "title": "Primary Care - Diabetes and CKD Follow-up",
                "date": "2026-03-05",
                "author": "Dr. Lena Brooks",
                "text": (
                    "Chief Complaint: diabetes follow-up. 63-year-old man with type 2 diabetes "
                    "mellitus for 14 years, stage 3b chronic kidney disease, hypertension, and "
                    "obesity. Reports polyuria and fatigue; denies chest pain. Medications: metformin "
                    "1000 mg BID, empagliflozin 10 mg daily, lisinopril 20 mg daily, atorvastatin "
                    "40 mg. Exam: BP 148/86, BMI 34, 1+ bilateral edema. Labs: HbA1c 9.4%, glucose "
                    "312 mg/dL, creatinine 2.1 mg/dL, eGFR 32 mL/min/1.73m2, potassium 5.4 mmol/L, "
                    "urine albumin-creatinine ratio 480 mg/g. Assessment: uncontrolled type 2 "
                    "diabetes, CKD stage 3b with albuminuria, hyperkalemia, hypertension not at goal. "
                    "Plan: add semaglutide, reduce metformin to 500 mg BID, stop lisinopril "
                    "temporarily, low-potassium diet, repeat BMP in 2 weeks, diabetes education "
                    "referral."
                ),
            }
        ],
        "labs": [
            {"test_name": "HbA1c", "value": 9.4, "unit": "%", "reference_range": "4.0-5.6", "date": "2026-03-05"},
            {"test_name": "Glucose", "value": 312, "unit": "mg/dL", "reference_range": "70-140", "date": "2026-03-05"},
            {"test_name": "Creatinine", "value": 2.1, "unit": "mg/dL", "reference_range": "0.6-1.3", "date": "2026-03-05"},
            {"test_name": "eGFR", "value": 32, "unit": "mL/min/1.73m2", "reference_range": ">60", "date": "2026-03-05"},
            {"test_name": "Potassium", "value": 5.4, "unit": "mmol/L", "reference_range": "3.5-5.1", "date": "2026-03-05"},
            {"test_name": "Urine albumin-creatinine ratio", "value": 480, "unit": "mg/g", "reference_range": "<30", "date": "2026-03-05"},
            {"test_name": "Sodium", "value": 137, "unit": "mmol/L", "reference_range": "135-145", "date": "2026-03-05"},
        ],
    },
    {
        "mrn": " 100788 ",
        "first_name": "Priya",
        "last_name": "Raman",
        "gender": "F ",
        "phone": "917-555-0110",
        "address": {"line": "250 E 63rd St", "city": "New York", "state": "NY", "postal_code": "10065"},
        "encounters": [{"class": "EMER", "reason": "Asthma exacerbation", "start": "2026-01-12"}],
        "documents": [],
        "labs": [],
    },
    {
        "mrn": "100788",
        "first_name": "Priya",
        "last_name": "Raman",
        "dob": "11/22/1990",
        "gender": "Female",
        "encounters": [{"class": "EMER", "reason": "Asthma exacerbation", "start": "2026-01-12"}],
        "documents": [
            {
                "type": "Progress Note",
                "title": "Emergency Department Note - Asthma Exacerbation",
                "date": "2026-01-12",
                "author": "Dr. Samuel Ortiz",
                "text": (
                    "Emergency Department Note. Chief Complaint: wheezing and shortness of breath "
                    "for two days after a viral upper respiratory infection. 35-year-old woman with "
                    "moderate persistent asthma. She used albuterol every two hours overnight without "
                    "relief. Exam: RR 26, SpO2 93% on room air, diffuse expiratory wheezes, speaking "
                    "in short sentences. Peak flow 280 L/min (baseline 480). Chest radiograph: "
                    "hyperinflation without focal consolidation or pneumothorax. Treatment: "
                    "albuterol-ipratropium nebulizers x3, oral prednisone 60 mg, magnesium sulfate 2 g "
                    "IV. Reassessed after treatment: SpO2 97%, peak flow 420 L/min, wheezing "
                    "decreased. Assessment: moderate asthma exacerbation triggered by viral URI. Plan: "
                    "prednisone 50 mg daily x5 days, continue inhaled corticosteroid-formoterol, asthma "
                    "action plan reviewed, return precautions."
                ),
            },
            {
                "type": "Imaging Report",
                "title": "Chest Radiograph - PA and Lateral",
                "date": "2026-01-12",
                "author": "Dr. M. Rosen",
                "text": (
                    "Chest radiograph PA and lateral. Lungs are hyperinflated with flattened "
                    "diaphragms. No focal consolidation, pleural effusion, or pneumothorax. "
                    "Cardiomediastinal silhouette normal. Impression: hyperinflation consistent with "
                    "obstructive airway disease; no acute infiltrate."
                ),
            },
        ],
        "labs": [
            {"test_name": "WBC", "value": 9.8, "unit": "10^3/uL", "reference_range": "4.0-11.0", "date": "2026-01-12"},
            {"test_name": "Hemoglobin", "value": 13.2, "unit": "g/dL", "reference_range": "12.0-17.5", "date": "2026-01-12"},
            {"test_name": "Sodium", "value": 138, "unit": "mmol/L", "reference_range": "135-145", "date": "2026-01-12"},
            {"test_name": "Potassium", "value": 3.9, "unit": "mmol/L", "reference_range": "3.5-5.1", "date": "2026-01-12"},
        ],
    },
    {
        "mrn": "MRN 101002",
        "first_name": "James",
        "last_name": "O'Connor",
        "dob": "04/18/1975",
        "gender": "1",
        "phone": "(646) 555-0121",
        "address": {"line": "701 Fort Washington Ave", "city": "New York", "state": "New York", "postal_code": "10040"},
        "encounters": [{"class": "Inpatient", "reason": "Sepsis secondary to community-acquired pneumonia", "start": "2026-02-02", "end": "2026-02-07"}],
        "documents": [
            {
                "type": "Progress Note",
                "title": "Admission History and Physical",
                "date": "2026-02-02",
                "author": "Dr. Hannah Lee",
                "text": (
                    "Chief Complaint: fever, productive cough, and confusion for two days. "
                    "50-year-old man with hypertension and former tobacco use. Temperature 39.1 C, "
                    "BP 92/58, HR 118, RR 28, SpO2 89% on room air. Examination: ill-appearing, "
                    "crackles and egophony at the right base. Labs: WBC 22.4, lactate 4.2 mmol/L, "
                    "procalcitonin 8.1 ng/mL, creatinine 1.6, sodium 132. Chest CT: dense right lower "
                    "lobe consolidation with air bronchograms and a small parapneumonic effusion. "
                    "Blood cultures drawn. Assessment: sepsis secondary to community-acquired "
                    "pneumonia, likely Streptococcus pneumoniae. Plan: ceftriaxone and azithromycin, "
                    "30 mL/kg crystalloid bolus, supplemental oxygen, ICU consultation, repeat lactate."
                ),
            },
            {
                "type": "Imaging Report",
                "title": "CT Chest with Contrast",
                "date": "2026-02-02",
                "author": "Dr. M. Rosen",
                "text": JAMES_CT,
            },
        ],
        "labs": [
            {"test_name": "WBC", "value": 22.4, "unit": "10^3/uL", "reference_range": "4.0-11.0", "date": "2026-02-02"},
            {"test_name": "Lactate", "value": 4.2, "unit": "mmol/L", "reference_range": "0.5-2.0", "date": "2026-02-02"},
            {"test_name": "Procalcitonin", "value": 8.1, "unit": "ng/mL", "reference_range": "<0.1", "date": "2026-02-02"},
            {"test_name": "Creatinine", "value": 1.6, "unit": "mg/dL", "reference_range": "0.6-1.3", "date": "2026-02-02"},
            {"test_name": "Sodium", "value": 132, "unit": "mmol/L", "reference_range": "135-145", "date": "2026-02-02"},
            {"test_name": "Hemoglobin", "value": 13.8, "unit": "g/dL", "reference_range": "12.0-17.5", "date": "2026-02-02"},
            {"test_name": "Platelets", "value": 210, "unit": "10^3/uL", "reference_range": "150-400", "date": "2026-02-02"},
        ],
    },
    {
        "mrn": "101230",
        "first_name": "Sofia",
        "last_name": "Alvarez",
        "dob": "09/09/1988",
        "gender": "female",
        "phone": "917-555-0110",
        "encounters": [{"class": "AMB", "reason": "Migraine follow-up", "start": "2026-03-18"}],
        "documents": [
            {
                "type": "Consultation",
                "title": "Neurology Consultation - Headache",
                "date": "2026-03-18",
                "author": "Dr. Ravi Shah",
                "text": (
                    "Chief Complaint: recurrent throbbing headaches for six months. 37-year-old woman "
                    "with a history of episodic migraine without aura. Headaches occur 3-4 times per "
                    "month, unilateral left temporal, associated with photophobia, nausea, and "
                    "occasional visual shimmering lasting 20 minutes. No fever, neck stiffness, "
                    "weakness, or speech difficulty. Neurological examination normal, including "
                    "fundoscopy and cranial nerves. MRI brain without contrast: no acute infarct, "
                    "mass, or hemorrhage; a few nonspecific white matter foci. Assessment: migraine "
                    "without aura, stable; medication overuse headache not present. Plan: continue "
                    "sumatriptan as needed, start propranolol 20 mg BID for prevention, headache "
                    "diary, follow up in 8 weeks."
                ),
            },
            {
                "type": "Imaging Report",
                "title": "MRI Brain without Contrast",
                "date": "2026-03-16",
                "author": "Dr. M. Rosen",
                "text": (
                    "MRI brain without and with diffusion-weighted imaging. No acute infarction, "
                    "intracranial hemorrhage, mass lesion, or midline shift. Ventricles and sulci are "
                    "age-appropriate. A few tiny nonspecific subcortical white matter "
                    "hyperintensities. Orbits and paranasal sinuses unremarkable. Impression: no acute "
                    "intracranial abnormality."
                ),
            },
        ],
        "labs": [
            {"test_name": "Hemoglobin", "value": 13.4, "unit": "g/dL", "reference_range": "12.0-17.5", "date": "2026-03-18"},
            {"test_name": "TSH", "value": 1.8, "unit": "mIU/L", "reference_range": "0.4-4.0", "date": "2026-03-18"},
            {"test_name": "Sodium", "value": 140, "unit": "mmol/L", "reference_range": "135-145", "date": "2026-03-18"},
            {"test_name": "Glucose", "value": 92, "unit": "mg/dL", "reference_range": "70-140", "date": "2026-03-18"},
        ],
    },
    {
        "mrn": "101455",
        "first_name": "Henry",
        "last_name": "Whitfield",
        "dob": "12/01/1936",
        "gender": "male",
        "phone": "555-CALL",
        "encounters": [{"class": "AMB", "reason": "Prostate cancer follow-up", "start": "2026-04-08"}],
        "documents": [
            {
                "type": "Consultation",
                "title": "Oncology Consultation - Rising PSA",
                "date": "2026-04-08",
                "author": "Dr. Priya Desai",
                "text": (
                    "Oncology Consultation. Chief Complaint: rising PSA and bone pain. 89-year-old man "
                    "with a history of localized prostate adenocarcinoma diagnosed in 2019, treated "
                    "with external beam radiation. Current PSA 18.4 ng/mL, up from 4.2 six months ago. "
                    "Reports new dull lower back pain. Examination: ECOG 1, no focal neurologic "
                    "deficit, back tender to palpation at L3. Bone scan: increased uptake in the L3 "
                    "vertebral body and right posterior iliac crest, concerning for osseous metastases. "
                    "Labs: hemoglobin 10.8 g/dL, creatinine 1.2, alkaline phosphatase 168 U/L, PSA "
                    "18.4. Assessment: biochemically recurrent prostate cancer with suspected bone "
                    "metastases. Plan: CT abdomen/pelvis, MRI lumbar spine, start androgen deprivation "
                    "therapy with leuprolide, bone health optimization with denosumab, palliative care "
                    "referral, follow-up in 4 weeks."
                ),
            },
            {
                "type": "Imaging Report",
                "title": "Whole-Body Bone Scan",
                "date": "2026-04-07",
                "author": "Dr. M. Rosen",
                "text": (
                    "Whole-body bone scan. Focal increased radiotracer uptake in the L3 vertebral body "
                    "and right posterior iliac crest. No other suspicious foci. Kidneys and bladder "
                    "visualize normally. Impression: two foci suspicious for osseous metastatic "
                    "disease; correlate with MRI."
                ),
            },
        ],
        "labs": [
            {"test_name": "PSA", "value": 18.4, "unit": "ng/mL", "reference_range": "<4.0", "date": "2026-04-08"},
            {"test_name": "Hemoglobin", "value": 10.8, "unit": "g/dL", "reference_range": "12.0-17.5", "date": "2026-04-08"},
            {"test_name": "Alkaline Phosphatase", "value": 168, "unit": "U/L", "reference_range": "44-147", "date": "2026-04-08"},
            {"test_name": "Creatinine", "value": 1.2, "unit": "mg/dL", "reference_range": "0.6-1.3", "date": "2026-04-08"},
        ],
    },
    {
        "mrn": "101678",
        "first_name": "Aisha",
        "last_name": "Nakamura",
        "dob": "03/27/1995",
        "gender": "F",
        "phone": "212-555-0155",
        "encounters": [{"class": "AMB", "reason": "Hypothyroidism follow-up", "start": "2026-02-25"}],
        "documents": [
            {
                "type": "Progress Note",
                "title": "Primary Care - Hypothyroidism Follow-up",
                "date": "2026-02-25",
                "author": "Dr. Lena Brooks",
                "text": (
                    "Chief Complaint: fatigue and cold intolerance. 30-year-old woman with Hashimoto "
                    "thyroiditis on levothyroxine 50 mcg daily. Reports fatigue, constipation, and dry "
                    "skin for three months. Examination: bradycardia 58 bpm, dry skin, delayed "
                    "relaxation of deep tendon reflexes, no goiter. Labs: TSH 8.9 mIU/L, free T4 0.7 "
                    "ng/dL, hemoglobin 11.9, vitamin D 22 ng/mL. Assessment: inadequately treated "
                    "primary hypothyroidism, vitamin D insufficiency. Plan: increase levothyroxine to "
                    "75 mcg daily, recheck TSH in 6 weeks, vitamin D3 2000 IU daily, counsel on "
                    "adherence."
                ),
            }
        ],
        "labs": [
            {"test_name": "TSH", "value": 8.9, "unit": "mIU/L", "reference_range": "0.4-4.0", "date": "2026-02-25"},
            {"test_name": "Free T4", "value": 0.7, "unit": "ng/dL", "reference_range": "0.8-1.8", "date": "2026-02-25"},
            {"test_name": "Vitamin D", "value": 22, "unit": "ng/mL", "reference_range": "30-100", "date": "2026-02-25"},
            {"test_name": "Hemoglobin", "value": 11.9, "unit": "g/dL", "reference_range": "12.0-17.5", "date": "2026-02-25"},
            {"test_name": "Sodium", "value": 139, "unit": "mmol/L", "reference_range": "135-145", "date": "2026-02-25"},
        ],
    },
    {
        "mrn": "101899",
        "first_name": "Daniel",
        "last_name": "Okafor",
        "dob": "06/14/1969",
        "gender": "M",
        "phone": "718-555-0166",
        "encounters": [{"class": "AMB", "reason": "Cirrhosis decompensation", "start": "2026-03-29"}],
        "documents": [
            {
                "type": "Consultation",
                "title": "Hepatology Consultation - Decompensated Cirrhosis",
                "date": "2026-03-29",
                "author": "Dr. Miriam Goldstein",
                "text": (
                    "Hepatology Consultation. Chief Complaint: abdominal distension and fatigue. "
                    "56-year-old man with alcohol-associated cirrhosis, portal hypertension, and prior "
                    "variceal bleed. Reports increasing abdominal girth over three weeks and early "
                    "satiety. Examination: scleral icterus, spider angiomata, tense ascites, no hepatic "
                    "encephalopathy. Labs: platelets 78, total bilirubin 3.4 mg/dL, INR 1.9, albumin "
                    "2.6 g/dL, creatinine 1.1, sodium 129, AST 68, ALT 42. CT abdomen: nodular "
                    "cirrhotic liver, moderate ascites, splenomegaly, patent portal vein, no "
                    "hepatocellular carcinoma. Assessment: decompensated cirrhosis with ascites and "
                    "coagulopathy; hyponatremia. Plan: therapeutic paracentesis with albumin, "
                    "spironolactone 100 mg and furosemide 40 mg daily, sodium restriction, band "
                    "surveillance in 2 weeks, transplant evaluation referral, alcohol cessation "
                    "counseling."
                ),
            },
            {
                "type": "Imaging Report",
                "title": "CT Abdomen and Pelvis with Contrast",
                "date": "2026-03-28",
                "author": "Dr. M. Rosen",
                "text": (
                    "CT abdomen and pelvis with contrast. Cirrhotic liver with nodular contour and "
                    "moderate ascites. Spleen enlarged at 15 cm. Portal and splenic veins patent. No "
                    "focal hepatic lesion. Small hiatal hernia. Impression: cirrhosis with portal "
                    "hypertension and moderate ascites; no hepatocellular carcinoma."
                ),
            },
        ],
        "labs": [
            {"test_name": "Platelets", "value": 78, "unit": "10^3/uL", "reference_range": "150-400", "date": "2026-03-29"},
            {"test_name": "Total Bilirubin", "value": 3.4, "unit": "mg/dL", "reference_range": "0.1-1.2", "date": "2026-03-29"},
            {"test_name": "INR", "value": 1.9, "unit": "ratio", "reference_range": "0.8-1.2", "date": "2026-03-29"},
            {"test_name": "Albumin", "value": 2.6, "unit": "g/dL", "reference_range": "3.5-5.0", "date": "2026-03-29"},
            {"test_name": "Sodium", "value": 129, "unit": "mmol/L", "reference_range": "135-145", "date": "2026-03-29"},
            {"test_name": "AST", "value": 68, "unit": "U/L", "reference_range": "10-40", "date": "2026-03-29"},
            {"test_name": "ALT", "value": 42, "unit": "U/L", "reference_range": "7-56", "date": "2026-03-29"},
            {"test_name": "Creatinine", "value": 1.1, "unit": "mg/dL", "reference_range": "0.6-1.3", "date": "2026-03-29"},
        ],
    },
    {
        "mrn": "101899",
        "first_name": "Daniel",
        "last_name": "Okafo",
        "dob": "06/15/1969",
        "gender": "M",
        "phone": "718-555-0166",
        "documents": [],
        "labs": [],
    },
    {
        "mrn": "102120",
        "first_name": "Grace",
        "last_name": "Lindqvist",
        "gender": "F",
        "phone": "347-555-0188",
        "encounters": [{"class": "EMER", "reason": "COPD exacerbation", "start": "2026-01-30"}],
        "documents": [
            {
                "type": "Progress Note",
                "title": "Emergency Department Note - COPD Exacerbation",
                "date": "2026-01-30",
                "author": "Dr. Samuel Ortiz",
                "text": (
                    "Emergency Department Note. Chief Complaint: increased sputum and breathlessness. "
                    "72-year-old woman with severe COPD (FEV1 38% predicted), former smoker, and "
                    "hypertension. Reports three days of purulent sputum and worsening dyspnea. "
                    "Examination: RR 24, SpO2 88% on room air, barrel chest, diffuse wheezes and coarse "
                    "crackles. Chest radiograph: hyperinflation and increased bibasilar markings, no "
                    "consolidation or pneumothorax. Labs: WBC 12.8, hemoglobin 14.2, sodium 136, "
                    "potassium 4.1, bicarbonate 32. Treatment: controlled oxygen to SpO2 90-92%, "
                    "ipratropium-albuterol nebulizers, prednisone 40 mg, doxycycline, noninvasive "
                    "ventilation considered but not required. Assessment: COPD exacerbation, likely "
                    "bacterial bronchitis. Plan: prednisone taper, antibiotic course, inhaler technique "
                    "review, pulmonary rehab referral."
                ),
            },
            {
                "type": "Imaging Report",
                "title": "Chest Radiograph - COPD Exacerbation",
                "date": "2026-01-30",
                "author": "Dr. M. Rosen",
                "text": (
                    "Chest radiograph. Hyperinflated lungs with flattened hemidiaphragms. Increased "
                    "bibasilar bronchovascular markings. No focal consolidation, pleural effusion, or "
                    "pneumothorax. Impression: changes of COPD without acute infiltrate."
                ),
            },
        ],
        "labs": [
            {"test_name": "WBC", "value": 12.8, "unit": "10^3/uL", "reference_range": "4.0-11.0", "date": "2026-01-30"},
            {"test_name": "Hemoglobin", "value": 14.2, "unit": "g/dL", "reference_range": "12.0-17.5", "date": "2026-01-30"},
            {"test_name": "Sodium", "value": 136, "unit": "mmol/L", "reference_range": "135-145", "date": "2026-01-30"},
            {"test_name": "Potassium", "value": 4.1, "unit": "mmol/L", "reference_range": "3.5-5.1", "date": "2026-01-30"},
            {"test_name": "Bicarbonate", "value": 32, "unit": "mmol/L", "reference_range": "22-29", "date": "2026-01-30"},
        ],
    },
    {
        "mrn": "102344",
        "first_name": "Robert",
        "last_name": "Chen",
        "dob": "08/05/1981",
        "gender": "M",
        "phone": "646-555-0129",
        "encounters": [{"class": "EMER", "reason": "Renal colic", "start": "2026-05-10"}],
        "documents": [
            {
                "type": "Progress Note",
                "title": "Emergency Department Note - Renal Colic",
                "date": "2026-05-10",
                "author": "Dr. Samuel Ortiz",
                "text": (
                    "Emergency Department Note. Chief Complaint: severe right flank pain radiating to "
                    "the groin for six hours, associated with nausea and hematuria. 44-year-old man "
                    "with no prior kidney stones. Temperature 37.2 C, BP 138/84, HR 96. Examination: "
                    "right costovertebral angle tenderness, no peritoneal signs. Urinalysis: 3+ blood, "
                    "1+ protein, negative nitrite and leukocyte esterase. CT abdomen/pelvis without "
                    "contrast: 6 mm obstructing stone at the right ureterovesical junction with mild "
                    "right hydronephrosis. Labs: creatinine 1.0, WBC 12.1, sodium 139. Assessment: "
                    "right ureteral stone with mild obstruction; no signs of infection. Plan: "
                    "ketorolac, tamsulosin 0.4 mg daily, strain urine, urology follow-up, return for "
                    "fever or uncontrolled pain."
                ),
            },
            {
                "type": "Imaging Report",
                "title": "CT Abdomen and Pelvis without Contrast",
                "date": "2026-05-10",
                "author": "Dr. M. Rosen",
                "text": (
                    "CT abdomen and pelvis without contrast. A 6 mm calcified stone is seen at the "
                    "right ureterovesical junction with mild proximal right hydronephrosis. No "
                    "left-sided stone. No appendicitis or diverticulitis. Impression: obstructing right "
                    "distal ureteral calculus with mild hydronephrosis."
                ),
            },
        ],
        "labs": [
            {"test_name": "Creatinine", "value": 1.0, "unit": "mg/dL", "reference_range": "0.6-1.3", "date": "2026-05-10"},
            {"test_name": "WBC", "value": 12.1, "unit": "10^3/uL", "reference_range": "4.0-11.0", "date": "2026-05-10"},
            {"test_name": "Sodium", "value": 139, "unit": "mmol/L", "reference_range": "135-145", "date": "2026-05-10"},
        ],
    },
    {
        "mrn": "102566",
        "first_name": "Lena",
        "last_name": "Petrova",
        "dob": "05/16/2001",
        "gender": "",
        "phone": "646-555-0133",
        "encounters": [{"class": "EMER", "reason": "Acute appendicitis", "start": "2026-04-02"}],
        "documents": [
            {
                "type": "Progress Note",
                "title": "Emergency Department Note - Abdominal Pain",
                "date": "2026-04-02",
                "author": "Dr. Hannah Lee",
                "text": (
                    "Emergency Department Note. Chief Complaint: periumbilical pain migrating to the "
                    "right lower quadrant for 18 hours, with nausea and low-grade fever. 24-year-old "
                    "woman, no significant past medical history. Temperature 38.0 C, HR 102, BP "
                    "118/72. Examination: tenderness at McBurney point with guarding and rebound. Labs: "
                    "WBC 15.2, hemoglobin 12.8, sodium 137. Ultrasound abdomen: enlarged "
                    "non-compressible appendix measuring 9 mm with surrounding fat stranding and a small "
                    "amount of free fluid; no appendicolith. Assessment: acute uncomplicated "
                    "appendicitis. Plan: NPO, IV fluids, ceftriaxone and metronidazole, general surgery "
                    "consultation for laparoscopic appendectomy."
                ),
            },
            {
                "type": "Imaging Report",
                "title": "Ultrasound Abdomen and Pelvis",
                "date": "2026-04-02",
                "author": "Dr. M. Rosen",
                "text": (
                    "Ultrasound abdomen and pelvis. The appendix is enlarged, non-compressible, and "
                    "measures 9 mm in diameter with surrounding hyperechoic fat stranding. A small "
                    "amount of free fluid is present in the right lower quadrant. No abscess. "
                    "Impression: sonographic findings compatible with acute appendicitis."
                ),
            },
        ],
        "labs": [
            {"test_name": "WBC", "value": 15.2, "unit": "10^3/uL", "reference_range": "4.0-11.0", "date": "2026-04-02"},
            {"test_name": "Hemoglobin", "value": 12.8, "unit": "g/dL", "reference_range": "12.0-17.5", "date": "2026-04-02"},
            {"test_name": "Sodium", "value": 137, "unit": "mmol/L", "reference_range": "135-145", "date": "2026-04-02"},
        ],
    },
    {
        "mrn": "102789",
        "first_name": "Omar",
        "last_name": "Haddad",
        "dob": "10/21/1943",
        "gender": "M",
        "phone": "212-555-0177",
        "encounters": [{"class": "EMER", "reason": "Acute ischemic stroke", "start": "2026-06-05"}],
        "documents": [
            {
                "type": "Progress Note",
                "title": "Stroke Neurology Note - Code Stroke",
                "date": "2026-06-05",
                "author": "Dr. Ravi Shah",
                "text": (
                    "Stroke Neurology Note. Chief Complaint: sudden right-sided weakness and difficulty "
                    "speaking. 82-year-old man with hypertension, hyperlipidemia, and atrial "
                    "fibrillation not on anticoagulation. Last known well 07:30; arrived 08:05. NIHSS 9 "
                    "for right facial droop, right arm and leg weakness, and mild expressive aphasia. CT "
                    "head without contrast: no hemorrhage; early ischemic changes not visualized. CT "
                    "angiography: occlusion of the left middle cerebral artery M2 segment. Labs: "
                    "glucose 118, INR 1.1, platelets 210, creatinine 1.3. Given alteplase at 08:40 after "
                    "blood pressure control. Admitted to the stroke unit. MRI brain confirmed acute left "
                    "MCA territory infarct. Assessment: acute ischemic stroke, left MCA M2 occlusion, "
                    "atrial fibrillation. Plan: stroke unit monitoring, dysphagia screening, start "
                    "anticoagulation after 24 hours, physical and speech therapy, echocardiogram."
                ),
            },
            {
                "type": "Imaging Report",
                "title": "MRI Brain with Diffusion-Weighted Imaging",
                "date": "2026-06-05",
                "author": "Dr. M. Rosen",
                "text": (
                    "MRI brain with diffusion-weighted imaging. Restricted diffusion in the left insular "
                    "cortex and left frontal operculum consistent with acute infarction in the left "
                    "middle cerebral artery territory. No hemorrhage. No mass effect. MRA demonstrates "
                    "occlusion of the left M2 segment. Impression: acute left MCA territory infarct."
                ),
            },
        ],
        "labs": [
            {"test_name": "Glucose", "value": 118, "unit": "mg/dL", "reference_range": "70-140", "date": "2026-06-05"},
            {"test_name": "INR", "value": 1.1, "unit": "ratio", "reference_range": "0.8-1.2", "date": "2026-06-05"},
            {"test_name": "Platelets", "value": 210, "unit": "10^3/uL", "reference_range": "150-400", "date": "2026-06-05"},
            {"test_name": "Creatinine", "value": 1.3, "unit": "mg/dL", "reference_range": "0.6-1.3", "date": "2026-06-05"},
            {"test_name": "LDL Cholesterol", "value": 142, "unit": "mg/dL", "reference_range": "<100", "date": "2026-06-05"},
        ],
    },
]

# ---------------------------------------------------------------------------
# Routine patients (to reach a realistic corpus of 50 patients)
# ---------------------------------------------------------------------------

ROUTINE_NAMES = [
    ("Alice", "Nguyen"), ("Benjamin", "Carter"), ("Camila", "Torres"), ("David", "Kim"),
    ("Elena", "Rossi"), ("Farah", "Ahmed"), ("Gabriel", "Silva"), ("Hannah", "Cohen"),
    ("Ibrahim", "Yusuf"), ("Julia", "Novak"), ("Kevin", "O'Brien"), ("Lucia", "Moreno"),
    ("Michael", "Zhang"), ("Nadia", "Hassan"), ("Oscar", "Lindgren"), ("Paula", "Ortiz"),
    ("Quentin", "Brooks"), ("Rachel", "Adler"), ("Samuel", "Osei"), ("Tanya", "Ivanova"),
    ("Umar", "Farouk"), ("Vanessa", "Duarte"), ("William", "Hayes"), ("Ximena", "Rojas"),
    ("Yosef", "Ben-Ami"), ("Zoe", "Caldwell"), ("Adrian", "Mensah"), ("Bianca", "Romano"),
    ("Carlos", "Mendez"), ("Diana", "Petrov"), ("Ethan", "Park"), ("Fatima", "Diallo"),
    ("George", "Papadopoulos"), ("Hana", "Sato"), ("Isaac", "Levi"), ("Jasmine", "Khan"),
    ("Lucas", "Ferreira"),
]

ROUTINE_TEMPLATES = [
    {
        "condition": "Hypertension follow-up",
        "complaint": "blood pressure follow-up",
        "assessment": "hypertension above goal",
        "plan": "increase lisinopril to 20 mg daily, home blood pressure log, low-sodium diet, recheck in 4 weeks",
        "labs": [("Sodium", 139, "mmol/L", "135-145"), ("Potassium", 4.2, "mmol/L", "3.5-5.1"), ("Creatinine", 1.0, "mg/dL", "0.6-1.3")],
    },
    {
        "condition": "Type 2 diabetes follow-up",
        "complaint": "diabetes follow-up",
        "assessment": "type 2 diabetes with HbA1c above goal",
        "plan": "continue metformin, add empagliflozin, diabetes education, repeat HbA1c in 3 months",
        "labs": [("HbA1c", 7.9, "%", "4.0-5.6"), ("Glucose", 168, "mg/dL", "70-140"), ("Creatinine", 0.9, "mg/dL", "0.6-1.3")],
    },
    {
        "condition": "Annual physical examination",
        "complaint": "routine annual physical",
        "assessment": "healthy adult with borderline LDL cholesterol",
        "plan": "diet and exercise counseling, repeat lipids in 6 months, age-appropriate screening",
        "labs": [("Hemoglobin", 13.8, "g/dL", "12.0-17.5"), ("Glucose", 96, "mg/dL", "70-140"), ("LDL Cholesterol", 112, "mg/dL", "<100")],
    },
    {
        "condition": "Upper respiratory infection",
        "complaint": "sore throat and nasal congestion for four days",
        "assessment": "viral upper respiratory infection",
        "plan": "supportive care, fluids, rest, return if fever or shortness of breath develops",
        "labs": [("WBC", 7.2, "10^3/uL", "4.0-11.0"), ("Sodium", 138, "mmol/L", "135-145")],
    },
    {
        "condition": "Mechanical low back pain",
        "complaint": "lower back pain after lifting",
        "assessment": "mechanical low back pain without red flags",
        "plan": "NSAIDs as needed, physical therapy referral, return precautions for weakness or incontinence",
        "labs": [("Hemoglobin", 14.1, "g/dL", "12.0-17.5"), ("Creatinine", 0.9, "mg/dL", "0.6-1.3")],
    },
    {
        "condition": "Hyperlipidemia follow-up",
        "complaint": "cholesterol follow-up",
        "assessment": "hyperlipidemia not at LDL goal",
        "plan": "increase atorvastatin to 40 mg nightly, repeat lipid panel in 8 weeks, continue Mediterranean diet",
        "labs": [("LDL Cholesterol", 156, "mg/dL", "<100"), ("HDL Cholesterol", 44, "mg/dL", ">40"), ("Triglycerides", 180, "mg/dL", "<150"), ("ALT", 32, "U/L", "7-56")],
    },
    {
        "condition": "Gastroesophageal reflux disease",
        "complaint": "heartburn and regurgitation",
        "assessment": "gastroesophageal reflux disease without alarm features",
        "plan": "omeprazole 20 mg daily before breakfast, avoid late meals, weight management, reassess in 6 weeks",
        "labs": [("Hemoglobin", 13.1, "g/dL", "12.0-17.5"), ("Sodium", 140, "mmol/L", "135-145")],
    },
    {
        "condition": "Knee osteoarthritis",
        "complaint": "right knee pain with activity",
        "assessment": "moderate right knee osteoarthritis",
        "plan": "physical therapy, topical diclofenac, weight management, consider intra-articular injection",
        "labs": [("CRP", 6, "mg/L", "<10"), ("Hemoglobin", 13.5, "g/dL", "12.0-17.5")],
        "imaging": (
            "Radiograph right knee. Moderate medial joint space narrowing, subchondral sclerosis, and "
            "small marginal osteophytes. No acute fracture or large effusion. Impression: moderate "
            "osteoarthritis of the right knee."
        ),
    },
    {
        "condition": "Anxiety follow-up",
        "complaint": "anxiety and poor sleep",
        "assessment": "generalized anxiety disorder, improving on therapy",
        "plan": "continue sertraline 50 mg daily, cognitive behavioral therapy, sleep hygiene, follow up in 6 weeks",
        "labs": [("TSH", 2.1, "mIU/L", "0.4-4.0"), ("Hemoglobin", 12.9, "g/dL", "12.0-17.5")],
    },
    {
        "condition": "Asthma follow-up",
        "complaint": "asthma control check",
        "assessment": "well-controlled asthma with normal peak flow",
        "plan": "continue inhaled corticosteroid-formoterol as needed, refill albuterol, asthma action plan reviewed",
        "labs": [("WBC", 6.8, "10^3/uL", "4.0-11.0"), ("Hemoglobin", 13.6, "g/dL", "12.0-17.5")],
    },
]


def _age(dob: date, on: date) -> int:
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


def build_routine_patients() -> list[dict]:
    patients: list[dict] = []
    start = date(2026, 6, 1)
    for index, (given, family) in enumerate(ROUTINE_NAMES):
        template = ROUTINE_TEMPLATES[index % len(ROUTINE_TEMPLATES)]
        visit = start + timedelta(days=index * 3)
        dob = date(1950 + (index * 7) % 45, 1 + (index % 12), 1 + (index % 27))
        age = _age(dob, visit)
        gender = "F" if index % 2 == 0 else "M"
        mrn = f"{103000 + index * 7:08d}"
        bp_systolic = 128 + (index * 5) % 28
        bp_diastolic = 76 + (index * 3) % 14
        note = (
            f"{template['condition']} visit. Chief Complaint: {template['complaint']}. "
            f"{age}-year-old {'woman' if gender == 'F' else 'man'} with a history of "
            f"{template['condition'].lower()}. Reports adherence to medications with no acute "
            f"distress. Examination: BP {bp_systolic}/{bp_diastolic}, HR {68 + index % 18}, "
            f"afebrile, cardiopulmonary examination unremarkable. "
            f"Assessment: {template['assessment']}. Plan: {template['plan']}."
        )
        documents = [
            {
                "type": "Progress Note",
                "title": f"Primary Care - {template['condition']}",
                "date": visit.isoformat(),
                "author": "Dr. Lena Brooks" if index % 3 else "Dr. Samuel Ortiz",
                "text": note,
            }
        ]
        if template.get("imaging") and index % 5 == 0:
            documents.append(
                {
                    "type": "Imaging Report",
                    "title": "Radiograph Right Knee",
                    "date": (visit - timedelta(days=1)).isoformat(),
                    "author": "Dr. M. Rosen",
                    "text": template["imaging"],
                }
            )
        labs = [
            {
                "test_name": test,
                "value": value,
                "unit": unit,
                "reference_range": ref,
                "date": visit.isoformat(),
            }
            for test, value, unit, ref in template["labs"]
        ]
        patients.append(
            {
                "mrn": mrn,
                "first_name": given,
                "last_name": family,
                "dob": dob.isoformat(),
                "gender": gender,
                "phone": f"212-555-{2000 + index:04d}",
                "email": f"{given.lower()}.{family.lower().replace(' ', '')}@example.org",
                "address": {
                    "line": f"{100 + index} Example Ave",
                    "city": "New York",
                    "state": "NY",
                    "postal_code": f"100{index % 40:02d}",
                },
                "encounters": [
                    {
                        "class": "AMB",
                        "reason": template["condition"],
                        "start": visit.isoformat(),
                    }
                ],
                "documents": documents,
                "labs": labs,
            }
        )
    return patients


# ---------------------------------------------------------------------------
# CSV fixtures
# ---------------------------------------------------------------------------

ROSTER_ROWS = [
    ["MRN-100234", "Eleanor", "Vance", "1948-02-11", "F", "+1 212-555-0142", "New York", "NY", "10019"],
    ["100455", "Marcus", "Bell", "1962-07-03", "M", "646-555-0198", "Brooklyn", "NY", "11217"],
    ["101230", "Sofia", "Alvarez", "1988-09-09", "F", "917-555-0110", "New York", "NY", "10065"],
    ["101899", "Daniel", "Okafo", "1969-06-15", "M", "718-555-0166", "Queens", "NY", "11375"],
    ["101455", "Henry", "Whitfield", "1936-12-01", "M", "555-CALL", "Boston", "MA", "02115"],
    ["102789", "Omar", "Haddad", "1943-10-21", "M", "212-555-0177", "New York", "NY", "10029"],
    ["102566", "Lena", "Petrova", "2001-05-16", "", "646-555-0133", "New York", "NY", "10003"],
]

LAB_ROWS = [
    # Duplicates of JSON labs (should be removed by content hash).
    ["MRN-100234", "Potassium", "6.8", "mmol/L", "3.5-5.1", "02/14/2026", "final"],
    ["MRN-100234", "Creatinine", "2.4", "mg/dL", "0.6-1.3", "2026-02-14", "final"],
    ["MRN-100234", "BNP", "1450", "pg/mL", "<100", "2026-02-14", "final"],
    ["MRN-100234", "Potassium", "6.8", "mmol/L", "3.5-5.1", "02/14/2026", "final"],
    # Mixed formats, missing ranges/units, non-numeric values.
    ["100455", "HGB", "11,8", "", "", "03/05/2026", "final"],
    ["100455", "LDL", "128", "mg/dL", "", "Mar 5, 2026", "final"],
    ["100455", "HDL", "38", "mg/dL", ">40", "2026-03-05", "final"],
    ["100455", "Triglycerides", "210", "mg/dL", "<150", "2026-03-05", "final"],
    ["mrn 100788", "Peak Flow", ">200", "L/min", "", "2026-01-12", "final"],
    ["101002", "Lactate", "2.8", "mmol/L", "0.5-2.0", "2026-02-03", "final"],
    ["101002", "WBC", "18.2", "10^3/uL", "4.0-11.0", "2026-02-03", "final"],
    ["101455", "PSA", "18.4", "ng/mL", "<4.0", "05-MAR-2026", "final"],
    ["101455", "Calcium", "9.1", "mg/dL", "8.6-10.2", "2026-04-08", "final"],
    ["101455", "Albumin", "3.2", "g/dL", "3.5-5.0", "2026-04-08", "final"],
    ["101455", "LDH", "240", "U/L", "", "2026-04-08", "final"],
    ["101678", "Free T4", "0.7", "ng/dL", "0.8-1.8", "Mar 5, 2026", "final"],
    ["101899", "Ammonia", "84", "umol/L", "15-45", "2026-03-29", "final"],
    ["101899", "Platelets", "78", "10^3/uL", "150-400", "2026-03-29", "final"],
    ["102120", "WBC", "12.8", "10^3/uL", "4.0-11.0", "13/45/2025", "final"],
    ["102120", "Bicarbonate", "32", "mmol/L", "22-29", "2026-01-30", "final"],
    ["102344", "Urinalysis", "3+ BLOOD", "", "", "2026-05-10", "final"],
    ["102344", "Creatinine", "1.0", "mg/dL", "0.6-1.3", "20260510", "final"],
    ["102344", "WBC", "12.1", "10^3/uL", "4.0-11.0", "2026-05-10", "final"],
    ["102566", "Platelets", "260", "10^3/uL", "150-400", "2026-04-02", "final"],
    ["102566", "Potassium", "", "mmol/L", "3.5-5.1", "2026-04-02", "final"],
    ["102789", "LDL Cholesterol", "142", "mg/dL", "<100", "2026-06-05", "final"],
    # Malformed rows: missing test name, missing identifier, bad value.
    ["MRN-100234", "", "5.0", "mmol/L", "", "2026-02-14", "final"],
    ["", "Sodium", "140", "mmol/L", "135-145", "2026-02-14", "final"],
    ["101678", "TSH", "n/a", "mIU/L", "0.4-4.0", "2026-02-25", "final"],
]

# ---------------------------------------------------------------------------
# Text notes fixtures
# ---------------------------------------------------------------------------

TEXT_NOTES = f"""=== NOTE ===
MRN: MRN-100234
Patient: Vance, Eleanor
DOB: 02/11/1948
Date: 02/19/2026
Type: Discharge Summary
Author: Dr. Anita Patel
Title: Discharge Summary - Acute Decompensated Heart Failure
---
{ELEANOR_DISCHARGE}

=== NOTE ===
MRN: MRN-101002
Patient: O'Connor, James
DOB: 18/04/1975
Date: 2026-02-03
Type: Progress Note
Author: Dr. Hannah Lee
Title: ED Follow-up Note - Sepsis
---
Patient is a 50-year-old man admitted with sepsis secondary to right lower lobe pneumonia.
Repeat lactate improved to 2.8 mmol/L after fluid resuscitation. Blood pressure 108/64 on
norepinephrine 4 mcg/min. WBC 18.2. Antibiotics continued with ceftriaxone and azithromycin.
Assessment: sepsis improving, pneumonia resolving. Plan: wean norepinephrine, repeat lactate in
6 hours, transfer from ICU when stable.

=== NOTE ===
MRN: 102120
Patient: Lindqvist, Grace
Date: 13/45/2025
Type: Scanned Note
Author: Dr. Samuel Ortiz
---
SCANNED DOCUMENT - OCR QUALITY FAIR
Emergency  department  visit  for  COPD  exacerbation.
Patient  reports  increased  sputum  and  wheeze.  SpO2  88%  on  room  air.
Chest  x-ray  shows  hyperinflation,  no  consolidation.  Treated  with  nebulizers
and  prednisone.  Discharged  home  in  stable  condition.

=== NOTE ===
MRN: A-7781
Patient: Haddad, Omar
DOB: 10/21/1943
Date: 2026-06-06
Type: Progress Note
Author: Dr. Ravi Shah
Title: Stroke Unit Daily Note (legacy MRN)
---
Stroke unit day 1. Right arm strength 3/5, right leg 4/5, mild expressive aphasia improving.
NIHSS 5. Blood pressure 146/82. No hemorrhagic transformation on repeat imaging. Started
aspirin; anticoagulation planned after 24 hours. Physical and speech therapy initiated.
Assessment: acute left MCA infarct, improving. Plan: continue stroke unit care.

=== NOTE ===
MRN: mrn 102566
Patient: Petrova, Lena
DOB: 05/16/2001
Date: 2026-04-02
Type: Progress Note
Author: Dr. Hannah Lee
Title: Surgical Consultation - Appendicitis
---
General surgery consultation. 24-year-old woman with ultrasound-confirmed acute appendicitis.
WBC 15.2, no perforation or abscess on imaging. Plan: laparoscopic appendectomy after
antibiotics, pre-operative checklist completed.

=== NOTE ===
MRN: UNKNOWN
Patient: Nakamura, Aisha
DOB: 03/27/1995
Date: 2026-03-04
Type: Progress Note
Author: Dr. Lena Brooks
Title: Telephone Follow-up - Thyroid
---
Telephone follow-up. Patient reports improved energy after increasing levothyroxine to 75 mcg
daily. No palpitations. Repeat TSH scheduled in two weeks. Continue current dose.

=== NOTE ===
MRN: 102344
Patient: Chen, Robert
DOB: 08/05/1981
Date: 2026-05-11
Type: Progress Note
Author: Dr. Samuel Ortiz
Title: Urology Follow-up - Kidney Stone
---
Urology follow-up after emergency department visit. Stone has not passed. Pain controlled with
ketorolac and tamsulosin. No fever. Plan: continue tamsulosin, strain urine, CT in two weeks if
no passage, return precautions reviewed.

=== NOTE ===
MRN: 100788
Patient: Raman, Priya
DOB: 11/22/1990
Date: 2026-01-12
Type: Progress Note
Author: Dr. Samuel Ortiz
Title: Asthma Discharge Instructions
---
Asthma discharge instructions reviewed. Peak flow improved to 420 L/min after nebulizers.
Continue prednisone 50 mg daily for five days and inhaled corticosteroid-formoterol. Avoid
triggers. Return for worsening wheeze, chest pain, or difficulty speaking.

=== NOTE ===
MRN: 101899
Patient: Okafor, Daniel
DOB: 14/06/1969
Date: 2026-03-29
Type: Progress Note
Author: Dr. Miriam Goldstein
Title: Paracentesis Note
---
Therapeutic paracentesis performed, 4.1 L of clear ascitic fluid removed with albumin
replacement. Post-procedure vitals stable. Specimen sent for cell count and culture.
Assessment: tense ascites due to decompensated cirrhosis. Plan: continue diuretics, sodium
restriction, transplant evaluation.

=== NOTE ===
MRN: MRN 100455
Patient: Bell, Marcus
DOB: 07/03/1962
Date: 2026-03-05
Type: Progress Note
Author: Dr. Lena Brooks
Title: Diabetes Education Note
---
Diabetes education completed. Reviewed carbohydrate counting, home glucose monitoring, and
sick-day rules. Patient will start semaglutide weekly. Hold lisinopril until repeat potassium.
Follow-up labs in two weeks.
"""


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    patients = CORE_PATIENTS + build_routine_patients()

    (RAW / "ehr_export.json").write_text(
        json.dumps({"exported_at": "2026-09-30T12:00:00Z", "patients": patients}, indent=2),
        encoding="utf-8",
    )

    with (RAW / "patient_roster.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["mrn", "first_name", "last_name", "dob", "gender", "phone", "city", "state", "zip"]
        )
        writer.writerows(ROSTER_ROWS)

    with (RAW / "labs_2026.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["mrn", "test_name", "value", "unit", "reference_range", "collected_at", "status"]
        )
        writer.writerows(LAB_ROWS)

    (RAW / "clinical_notes.txt").write_text(TEXT_NOTES, encoding="utf-8")

    print(f"Wrote {len(patients)} patients to {RAW}")
    print(f"  ehr_export.json: {len(CORE_PATIENTS)} rich + {len(ROUTINE_NAMES)} routine")
    print(f"  patient_roster.csv: {len(ROSTER_ROWS)} rows")
    print(f"  labs_2026.csv: {len(LAB_ROWS)} rows")
    print("  clinical_notes.txt: 10 notes")


if __name__ == "__main__":
    main()
