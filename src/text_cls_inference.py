import pickle
from utils import clean_text
import torch

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
embedding_model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # All token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Sum the embeddings
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    # Sum the mask
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # Mean pooling
    return sum_embeddings / sum_mask

def get_embeddings(texts):
    texts = ["query: " + text for text in texts]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    return embeddings.cpu().numpy()


# Load the model
with open('./model/text_classify/logistic_regression_model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)

# Load the label encoder
with open('./model/text_classify/label_encoder.pkl', 'rb') as f:
    le_loaded = pickle.load(f)

def cls_inference(text_inputs):
  # clean text
  inputs = clean_text(text_inputs, label=None)

  # embeding text
  emb_inputs = get_embeddings([inputs])

  # Predict labels
  predicted_labels_encoded = clf_loaded.predict( emb_inputs)

  # Decode labels
  predicted_labels = le_loaded.inverse_transform(predicted_labels_encoded)

  return predicted_labels


if __name__ == "__main__":
    inputs = "Education and activities Nakhon Sawan Rajabhat University (NSRU), Nakhon Sawan < May 2019 – June 2023 > (Bachelor’s Degree in Political Science program in Politics and Government) GPAX : 3.56 (2nd class honor)"

    result = cls_inference(inputs)
    print(result[0])
    print(type(result[0]))