from config import  ARTIFACTS_DIR, MODEL_NAME,VECTORIZER_NAME
import pickle

class ModelLoader:
    def __init__(self,  model_name: str, vectorizer_name: str):
        self.model_name = model_name
        self.model_path = ARTIFACTS_DIR / model_name
        self.vectorizer_path = ARTIFACTS_DIR / VECTORIZER_NAME
        self.vectorizer_name = vectorizer_name
        self.model = None
        self.vectorizer = None

    def load_model(self):
        """Loads the ML model from MLflow."""
        try:
            print(f"Loading model from {self.model_path}")

            # Load the model
            with self.model_path.open("rb") as f:
                self.model = pickle.load(f)

            print(f"Loading vectorizer from {self.vectorizer_path}")
            with self.vectorizer_path.open("rb") as f:
                self.vectorizer = pickle.load(f)


            return self.model, self.vectorizer

        except Exception as e:
            print(f"Error loading model: {e}")



# Example usage
if __name__ == "__main__":
    loader = ModelLoader(MODEL_NAME,VECTORIZER_NAME)
    model, vectorizer = loader.load_model()
    print(type(model))
    print(type(vectorizer))
    print(vectorizer.transform(['this is a test']))

