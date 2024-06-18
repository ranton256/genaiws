# Generative AI Workshop

Model Development by *Richard Anton, Principal Engineer, Snowflake* portion of  the [Primer on Generative AI in Gastroenterology Hands-On Workshop](https://www.asge.org/home/education-meetings/event-detail/2024/05/17/default-calendar/artificial-intelligence-in-gi) May 17, 2024.

## Setup

The complete setup instructions are available in [DDW GenAI Workshop Prerequisites.pdf](DDW GenAI Workshop Prerequisites.pdf).

They are also available at https://tinyurl.com/setupgenaiws 

The slides are available at https://tinyurl.com/genaiws-slides

### Setup in Brief

1. Download ZIP archive of workshop files: https://tinyurl.com/genaiws2024 
2. Download and run Miniconda installer https://docs.anaconda.com/free/miniconda/miniconda-install/ 
3. Download, unzip, and run Ollama installer from https://ollama.com/ 
4. Extract ZIP file and follow instructions for setup in the DDW GenAI Workshop Prerequisites.pdf file.
   1. Setup Python environment
   2. Open Jupyter setup notebook: [ddw_genai_workshop_setup.ipynb](ddw_genai_workshop_setup.ipynb).
   3. Follow instructions in the notebook to download models and setup HuggingFace.

This repo contains the source code and other files for the Interactive Hands-On Session  - Generative AI 

This workshop covers foundational models and applications in natural language processing, in context of the medical field in general and gastroenterology in specific.

### Setup Walkthrough Videos

There are video recordings of the setup for the workshop lab available.

- YouTube Playlist: https://www.youtube.com/playlist?list=PL1ZfNGUkCn1tZTFkknw4CxbM3jgESVbaw

- Setup part 1 Mac: https://youtu.be/JMtfXQPVQhg
- Setup part 1 Win: https://youtu.be/TgITVLP-HE0
- Setup part 2: https://youtu.be/BiKrTAdAFxE


### Topics Covered

* Introduction to generative AI and large language model (LLM) concepts such as tokenization, embedding, transformer models, 
* Techniques for adapting pretrained models to new data and use cases iIncluding transfer learning, fine-tuning, retrieval augmented generation (RAG), reinforcement learning, and prompt engineering.
* An introduction to several pre-trained models including Mistral AI Mistral 7B Instruct, HuggingFace Zephyr 7b Beta, Meta Llama3, Google Gemma, and Snowflake Arctic.
* Text embedding models and vector stores.
* Walkthrough of building a RAG system for chatting with clinical guidelines and related documents for the diagnosis and treatment of [Barrett's esophagus](https://en.wikipedia.org/wiki/Barrett%27s_esophagus).
* RAG concepts, advantages, limitations, components and the flow of data.
* Environment setup for AI/ML experimentation and tools using Ollama and HuggingFace from Streamlit apps and Jupyter notebooks.
* Next steps and suggestions on where to learn more.
* Bibliography of referenced tools, papers, and articles.

This repo also contains an example source for creating an LLM Agent app using Streamlit and Langchain as an additional example for participants to explare later.

