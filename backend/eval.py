from evaluate import AnimalEvaluator
if __name__ == '__main__':
    evaluator = AnimalEvaluator(
        model_path='../checkpoints/animal_cnn.pth',
        eval_dir='animal_dataset/eval_dataset'
    )
    evaluator.evaluate()