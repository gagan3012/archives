# ArchivesRAG

Here I will be explaining step by step the training of ArchivesRAG, an embedding model specific for the Archives Domain. 

## Query generation

The first step is to generate the queries that will be used to train the model. The queries are generated using the [GPT-4o](https://chatgpt.com/?model=gpt-4o) model, which is a variant of the GPT-4 model. The queries are generated using the following prompt:

```markdown
You are a smart and helpful assistant.
    Your mission is to write one text retrieval example for this task in JSON format. The JSON object must contain the following keys:
"query": a string, a random user search query specified by the retrieval task.
"positive": This will be provided to you. It is a list of strings, each representing a positive example of a document that should be retrieved by the search query.

Please adhere to the following guidelines:
- The "query" should be a random user search query.
- The "query" should be paragraph-based, in at least 10 words, understandable with some effort or ambiguity, and diverse in topic.
- Query should be strongly related to the "positive" examples given to you.
- Your input will be just the text which is to be considered as the positive examples.
```

Original Data: 
```json
{
    "text": "InterPARES Trust is an international research project funded by the Social Sciences and Humanities Research Council of Canada. The project is designed to investigate issues concerning digital records and data entrusted to the Internet. The project is a collaborative effort of researchers from many countries and disciplines. The project is designed to investigate issues concerning digital records and data entrusted to the Internet. The project is a collaborative effort of researchers from many countries and disciplines."
}
```

To generate the queries we use the following code:

```shell
python gen_query.py --input_file data.jsonl --output_file query.jsonl
```

Generated Query:
```json
{
    "query": "What is the InterPARES Trust project about?",
    "positive": [
        "InterPARES Trust is an international research project funded by the Social Sciences and Humanities Research Council of Canada."
    ]
}
```

## Mine Hard Negatives

Hard negatives is a widely used method to improve the quality of sentence embedding.
You can mine hard negatives following this command:

```bash
python mine_hn.py \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file query.jsonl \
--output_file query_minedHN.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 
```

- `input_file`: json data for finetuning. This script will retrieve top-k documents for each query,
and random sample negatives from the top-k documents (not including the positive documents).
- `output_file`: path to save JSON data with mined hard negatives for finetuning
- `negative_number`: the number of sampled negatives
- `range_for_sampling`: where to sample negative. For example, `2-100` means sampling `negative_number` negatives from top2-top200 documents. **You can set larger value to reduce the difficulty of negatives (e.g., set it `60-300` to sample negatives from top60-300 passages)**
- `candidate_pool`: The pool to retrieval. The default value is None, and this script will retrieve from the combination of all `neg` in `input_file`.
- `use_gpu_for_searching`: whether to use faiss-gpu to retrieve negatives.


## Training

To train the model we use the following command:

```
number_of_gpus=1

torchrun --nproc_per_node 1 \
run.py \
--output_dir archivesrag \
--model_name_or_path mixedbread-ai/mxbai-embed-large-v1 \
--train_data query_minedHN.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 32 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--save_steps 1000 \
--query_instruction_for_retrieval "" 
```

**some important arguments**:

- `per_device_train_batch_size`: batch size in training. In most of cases, larger batch size will bring stronger performance. You can expand it by enabling `--fp16`, `--deepspeed ./df_config.json` (df_config.json can refer to [ds_config.json](./ds_config.json)), `--gradient_checkpointing`, etc.
- `train_group_size`: the number of positive and negatives for a query in training.
There are always one positive, so this argument will control the number of negatives (#negatives=train_group_size-1).
Noted that the number of negatives should not be larger than the numbers of negatives in data `"neg":List[str]`.
Besides the negatives in this group, the in-batch negatives also will be used in fine-tuning.
- `negatives_cross_device`: share the negatives across all GPUs. This argument will extend the number of negatives.
- `learning_rate`: select a appropriate for your model. Recommend 1e-5/2e-5/3e-5 for large/base/small-scale.
- `temperature`: It will influence the distribution of similarity scores. **Recommended value: 0.01-0.1.**
- `query_max_len`: max length for query. Please set it according the average length of queries in your data.
- `passage_max_len`: max length for passage. Please set it according the average length of passages in your data.
- `query_instruction_for_retrieval`: instruction for query, which will be added to each query. You also can set it `""` to add nothing to query.
- `use_inbatch_neg`: use passages in the same batch as negatives. Default value is True.
- `save_steps`: for setting how many training steps to save a checkpoint.

For more training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)


## Usage

To use the model for retrieval we use the following code:

```python

from datasets import load_dataset

dataset = load_dataset("json", data_files="{path_to_your_data}.jsonl") # Load your data here

from sentence_transformers import SentenceTransformer
ST = SentenceTransformer("ArchivesRAG") # Load the model

def embed(batch):
    """
    adds a column to the dataset called 'embeddings'
    """
    # or you can combine multiple columns here
    # For example the title and the text
    information = batch["text"]
    return {"embeddings" : ST.encode(information)}

dataset = dataset.map(embed,batched=True,batch_size=16)

data = dataset["train"]
data = data.add_faiss_index("embeddings")

def search(query: str, k: int = 3 ):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples( # retrieve results
        "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
        k=k # get only top k results
    )
    return scores, retrieved_examples

query = "What is the InterPARES Trust project about?"
scores, retrieved_examples = search(query)
print(retrieved_examples)

```

## Usage with LLama 3

To use the model with LLama 3 we use the following code:

```python
from vllm import LLM, SamplingParams
from transfromers import AutoTokenizer

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def format_prompt(prompt,retrieved_documents,k):
    """using the retrieved documents we will prompt the model to generate our responses"""
    PROMPT = f"Question:{prompt}\nContext:"
    for idx in range(k) :
        PROMPT+= f"{retrieved_documents['text'][idx]}\n"
    return PROMPT

SYSTEM_MESSAGE = """You are a helpful assistant for answering questions. Your primary focus is to assist in studies relating to Interpares Itrust AI.
InterPARES Trust AI (2021-2026) is a multi-national interdisciplinary project aiming to design, develop, and leverage Artificial Intelligence to support the ongoing availability and accessibility of trustworthy public records by forming a sustainable, ongoing partnership producing original research, training students and other highly qualified personnel (HQP), and generating a virtuous circle between academia, archival institutions, government records professionals, and industry, a feedback loop reinforcing the knowledge and capabilities of each party.
Use the following context elements to answer the question.
"""

k = 1 # number of retrieved documents
scores , retrieved_documents = search(prompt, k)
formatted_prompt = format_prompt(prompt,retrieved_documents,k)
formatted_prompt = formatted_prompt[:2000] # to avoid GPU OOM
messages = [{"role":"system","content":SYSTEM_MESSAGE},{"role":"user","content":formatted_prompt}]

MODEL_ID = ""

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

input_text = [tokenizer.apply_chat_template(messages)]

llm = LLM(model=MODEL_ID)

output = llm.generate(input_text,sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```