from application.handle_training import Trainer
from config.handle_configurations import parse_args

if __name__ == "__main__":
    args = parse_args()
    Trainer(args)
