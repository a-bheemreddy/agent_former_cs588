# model1.py
import torch
import sys
import logging

def load_model():
    # Load model1 using the specific PyTorch version
    return "Hello there"

def process_input(model, input_data):
    # Perform model1 operations using the input data
    output_data = "MODEL OUTPUT"
    return output_data

if __name__ == "__main__":
    model = load_model()
    print("Model1 loaded with PyTorch version", torch.__version__)
    sys.stdout.flush()
    # print("we here")
    # create logger for file logging.txt
    logger = logging.getLogger('logging')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('logging.txt')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    # Log some messages


    while True:
        # for line in sys.stdin:
        #     input_data = line.strip()
        #     if not input_data:
        #         break
        logger.info('Starting up')
        input_data = sys.stdin.readline().strip()
        logger.info(f'Have input data {input_data}')
        #print(input_data)
        if not input_data:
            logger.info('ERROR BREAKNG BREAKING')
            break
        output_data = process_input(model, input_data)
        logger.info(f'Have output data {output_data}')
        print(output_data)
        sys.stdout.flush()