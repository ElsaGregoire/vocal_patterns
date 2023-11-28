# This file is for saving, loading, and managing versions of models


def save_results(results, model_name):
    """Saves the results of a model to a file"""
    with open(f"models/{model_name}.txt", "w") as file:
        file.write(results)


def save_model(model, model_name):
    """Saves a model to a file"""
    model.save(f"models/{model_name}.h5")


def load_model(model_name):
    """Loads a model from a file"""
    return load_model(f"models/{model_name}.h5")
