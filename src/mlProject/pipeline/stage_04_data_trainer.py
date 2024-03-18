from mlProject.config.configuration import ConfigurationManager
from mlProject.component.data_trainer import ModelTrainer
from mlProject import logger


STAGE_NAME = "Data Trainer Stage"


class DataTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_trainer_config = config.get_model_trainer_config()
        data_trainer = ModelTrainer(config=data_trainer_config)  # noqa: E501
        data_trainer.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
