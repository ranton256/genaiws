# pip install vllm
# https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct?source=post_page-----c79f843f208c---------------------------------------
# python -m vllm.entrypoints.openai.api_server --model moonshotai/Kimi-Linear-48B-A3B-Instruct

from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

def get_kimi_linear_model():
    model_name = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

def main():
    load_dotenv()

    model, tokenizer = get_kimi_linear_model()

    # just demo it
    messages = [
        {"role": "system", "content": "You are a helpful assistant provided by Moonshot-AI."},
        {"role": "user", "content": "Is 123 a prime?"}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(inputs=input_ids, max_new_tokens=500)
    response = tokenizer.batch_decode(generated_ids)[0]
    print(response)

if __name__ == "__main__":
    main()

# citatin (bibtex)
"""
@misc{team2025kimi,
    title         = {Kimi Linear: An Expressive, Efficient Attention Architecture},
    author        = {Zhang, Yu  and Lin, Zongyu  and Yao, Xingcheng  and Hu, Jiaxi  and Meng, Fanqing  and Liu, Chengyin  and Men, Xin  and Yang, Songlin  and Li, Zhiyuan  and Li, Wentao  and Lu, Enzhe  and Liu, Weizhou  and Chen, Yanru  and Xu, Weixin  and Yu, Longhui  and Wang, Yejie  and Fan, Yu  and Zhong, Longguang  and Yuan, Enming  and Zhang, Dehao  and Zhang, Yizhi  and T. Liu, Y.  and Wang, Haiming  and Fang, Shengjun  and He, Weiran  and Liu, Shaowei  and Li, Yiwei  and Su, Jianlin  and Qiu, Jiezhong  and Pang, Bo  and Yan, Junjie  and Jiang, Zhejun  and Huang, Weixiao  and Yin, Bohong  and You, Jiacheng  and Wei, Chu  and Wang, Zhengtao  and Hong, Chao  and Chen, Yutian  and Chen, Guanduo  and Wang, Yucheng  and Zheng, Huabin  and Wang, Feng  and Liu, Yibo  and Dong, Mengnan  and Zhang, Zheng  and Pan, Siyuan  and Wu, Wenhao  and Wu, Yuhao  and Guan, Longyu  and Tao, Jiawen  and Fu, Guohong  and Xu, Xinran  and Wang, Yuzhi  and Lai, Guokun  and Wu, Yuxin  and Zhou, Xinyu  and Yang, Zhilin  and Du, Yulun},
    year          = {2025},
    eprint        = {2510.26692},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL}
}
"""

