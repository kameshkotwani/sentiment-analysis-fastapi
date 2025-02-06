import dagshub
from config import  MLFLOW_TRACKING_URI,MODEL_URI,VECTORIZER_URI,RUN_ID
import mlflow
from pathlib import Path
dagshub.init(
    repo_owner='kameshkotwani',
    repo_name='mlops-mini-project',
    mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
import pickle

class ModelLoader:
    def __init__(self, tracking_uri: str, run_id: str, model_name: str, vectorizer_name: str):
        self.tracking_uri = tracking_uri
        self.run_id = run_id
        self.model_name = model_name
        self.vectorizer_name = vectorizer_name
        self.artifacts_dir = Path("artifacts")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.vectorizer = None
        mlflow.set_tracking_uri(self.tracking_uri)

    def download_vectorizer(self):
        """Downloads and loads the vectorizer from MLflow."""
        try:
            print("Downloading vectorizer...")
            vectorizer_local_path = Path(
                mlflow.artifacts.download_artifacts(run_id=self.run_id, artifact_path= Path("artifacts") /  self.vectorizer_name))
            with vectorizer_local_path.open("rb") as f:
                self.vectorizer = pickle.load(f)
            print("Vectorizer loaded successfully!")
        except Exception as e:
            print(f"Error loading vectorizer: {e}")

    def load_model(self):
        """Loads the ML model from MLflow."""
        try:
            print("Loading model from MLflow...")
            model_uri = f"runs:/{self.run_id}/model"
            self.model = mlflow.pyfunc.load_model(model_uri)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

    def load(self):
        """Loads both model and vectorizer."""
        self.download_vectorizer()
        self.load_model()
        return self.model, self.vectorizer


# Example usage
if __name__ == "__main__":
    tracking_uri = MLFLOW_TRACKING_URI
    run_id = RUN_ID
    model_name = 'model'
    vectorizer_name = "vectorizer.pkl"

    loader = ModelLoader(tracking_uri, run_id, model_name, vectorizer_name)
    model, vectorizer = loader.load()

