from model.setting import Setting, Arguments
from model.ner_model.processor import Processor


def main(args, logger) -> None:
    processor = Processor(args)
    config = processor.model_setting()
    logger.info('Model Setting Complete')

    if args.train == 'True':
        logger.info('Start Training')

        for epoch in range(args.epochs):

            processor.train(epoch+1)

    if args.test == 'True':
        logger.info("Start Test")

        processor.test()

        processor.metric.print_size_of_model(config['model'])
        processor.metric.count_parameters(config['model'])

    if args.mask_post_training_data == 'True':
        processor.post_training()


if __name__ == '__main__':
    args, logger = Setting().run()
    main(args, logger)
