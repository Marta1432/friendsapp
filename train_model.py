from pycaret.clustering import *
import pandas as pd

# Wczytaj dane
df = pd.read_csv("welcome_survey_simple_v2.csv", sep=";")


# Inicjalizacja środowiska PyCaret
s = setup(df, session_id=123, normalize=True)

# Stwórz model
model = create_model("kmeans")

# Zapisz model
save_model(model, "welcome_survey_clustering_pipeline_v2")