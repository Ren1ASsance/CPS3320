from train import AnimalTrainer

if __name__ == '__main__':
    trainer = AnimalTrainer(
        train_dir='animal_dataset/dataset',
    )
    trainer.train()
