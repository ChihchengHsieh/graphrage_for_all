def get_questions_by_lesion(lesion: str):
    questions = [
        # f"What is {lesion}?", # don't need the first one for extending features.
        f"What are the symptoms associated with {lesion}?",
        f"What can cause {lesion}?",
        f"What are the patient’s symptoms that are relevant for {lesion}?",
        f"What are the relevant clinical signs for the etiological diagnosis of {lesion}?",
        f"What are the relevant laboratory data for the etiological diagnosis of {lesion}?",
        f"What are the relevant clinical characteristics for the etiological diagnosis of {lesion}?",
        f"What are the patient’s personal relevant history for the etiological diagnosis of {lesion}?",
    ]
    return questions
