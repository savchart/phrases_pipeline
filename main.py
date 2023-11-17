import argparse
from utils.process_data import DataProcessor
from utils.on_the_fly import OnTheFlyProcessor
from utils.prepare_w2v import PrepareModel


def main():
    parser = argparse.ArgumentParser(description="Batch and on-the-fly execution options.")
    parser.add_argument('--option', type=str, choices=['batch', 'on_the_fly'], help="Choose execution option.")
    args = parser.parse_args()

    model = PrepareModel()
    if not model.cut_model_path():
        model.cut_model()
    if not model.prune_model_path():
        model.prune_model()

    if args.option == 'batch':
        data_processor = DataProcessor()
        data_processor.batch_execution()
    elif args.option == 'on_the_fly':
        on_the_fly_parser = argparse.ArgumentParser()
        on_the_fly_parser.add_argument('string', type=str, help="Input string for on-the-fly execution.")
        on_the_fly_args = on_the_fly_parser.parse_args()

        on_the_fly_processor = OnTheFlyProcessor()
        result_phrase, similarity_score = on_the_fly_processor.on_the_fly_execution(on_the_fly_args.string)

        print(f"Input String: {on_the_fly_args.string}")
        print(f"Closest Match: {result_phrase}")
        print(f"Cosine Similarity Score: {similarity_score}")
    else:
        print("Invalid option. Choose 'batch' or 'on_the_fly'.")


if __name__ == "__main__":
    main()
