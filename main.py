from config import *
from src.data_utils import get_dataset, split_dataset
from src.model_utils import load_model, train_model, evaluate_model


if __name__ == '__main__':

    # Load model
    model, model_processor = load_model(model_name)

    # Load data
    train_dataset = get_dataset(data_path, "training", model_processor)
    validation_dataset = get_dataset(data_path, "validation", model_processor)
    validation_dataset, test_dataset = split_dataset(validation_dataset, test_size=test_size)

    # Train model
    classifier_trainer = train_model(model,
                                     model_processor,
                                     train_dataset=train_dataset,
                                     eval_dataset=validation_dataset,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     lr=lr,
                                     weight_decay=weight_decay)

    # Evaluate model
    evaluate_model(classifier_trainer, test_dataset, 'test')
