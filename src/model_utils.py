import numpy as np
from sklearn.metrics import accuracy_score
from pyeer.eer_info import get_eer_stats
from transformers import Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments, Wav2Vec2Processor
from src.plots_utils import plot_confusion_matrix


def load_model(model_name, num_labels=2):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model_processor = Wav2Vec2Processor.from_pretrained(model_name)
    return model, model_processor


def compute_metrics(eval_pred):
    scores, labels = eval_pred
    predictions = np.argmax(scores, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    true_scores = scores[:, 1]
    eer = get_eer_stats(labels, true_scores).eer
    return {"accuracy": accuracy, "eer": eer}


def train_model(model, model_processor, train_dataset, eval_dataset=None, batch_size=16, epochs=10, lr=1e-5, weight_decay=0.01):
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eer",
        learning_rate=lr,
        weight_decay=weight_decay,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model_processor,
        compute_metrics=compute_metrics,
    )
    print("Model device:", trainer.model.device)
    trainer.train()
    return trainer


def evaluate_model(trainer, dataset, test_name):
    eval_results = trainer.evaluate(dataset)
    print("Evaluation Results:", eval_results)
    plot_confusion_matrix(trainer, dataset, labels=['Fake', 'Real'], title=f'Confusion Matrix - {test_name} Set')