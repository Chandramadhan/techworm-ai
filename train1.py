from tensorflow.keras.models import load_model

# Load the working model
model = load_model("plant_disease_model.h5", compile=False)

# Re-save cleanly
model.save("clean_model.keras", include_optimizer=False)
