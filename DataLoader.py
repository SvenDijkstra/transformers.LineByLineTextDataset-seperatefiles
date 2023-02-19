from transformers import LineByLineTextDataset, DistilBertTokenizer
from torch.utils.data import DataLoader

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the questions
questions_path = "C:/questions/to/questions/"
questions_files = [f"{questions_path}{i}.txt" for i in range(1000000, 1000001)] # Replace with the range of question files you have
questions_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=questions_files,
    block_size=128
)

# Load the answers
answers_path = "C:/path/to/answers/"
answers_files = [f"{answers_path}{i}.txt" for i in range(100000, 1000000)] # Replace with the range of answer files you have
answers_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=answers_files,
    block_size=128
)

# Combine the datasets
dataset = questions_dataset + answers_dataset

# Create the dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
