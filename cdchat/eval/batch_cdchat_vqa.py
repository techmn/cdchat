import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from cdchat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cdchat.conversation import conv_templates, SeparatorStyle
from cdchat.model.builder import load_pretrained_model
from cdchat.utils import disable_torch_init
from cdchat.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import numpy as np


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, mm_projector_path=args.mm_projector_path)
    
    #questions=[]
    #questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    fid = open(args.question_file, 'r')
    questions = json.load(fid)
    fid.close()

        
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    ans_file = open(answers_file, "w")
    
    for i in tqdm(range(0,len(questions),args.batch_size)):
        input_batch=[]
        input_image_batch=[]
        count=i
        image_folder_A = []
        image_folder_B = []
        image_label_list = []
        batch_end = min(i + args.batch_size, len(questions))

             
        for j in range(i,batch_end):
            #image_file=questions[j]['image']
            #qs=questions[j]['text']
            image_file = questions[j]['img_id']
            print(image_file)
            qs = questions[j]['question']
            
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            input_batch.append(input_ids)

            #image = Image.open(os.path.join(args.image_folder, image_file))
            #image_folder.append(image)
            image_A = Image.open(os.path.join(args.image_folder, 'A', image_file)).convert('RGB')
            image_B = Image.open(os.path.join(args.image_folder, 'B', image_file)).convert('RGB')
            image_folder_label = os.path.join(args.image_folder, 'label')
            image_label = Image.open((os.path.join(image_folder_label, image_file)).strip()).convert('L')
            image_label = (torch.Tensor(1)*np.array(image_label)/255.).unsqueeze(0)

            image_tensor_A = image_processor.preprocess(image_A,crop_size ={'height': 448, 'width': 448},size = {'shortest_edge': 448}, return_tensors='pt')['pixel_values'][0]
            image_tensor_B = image_processor.preprocess(image_B,crop_size ={'height': 448, 'width': 448},size = {'shortest_edge': 448}, return_tensors='pt')['pixel_values'][0]

            image_tensor_A = image_tensor_A[[2,1,0], :, :]
            image_tensor_B = image_tensor_B[[2,1,0], :, :]

            image_folder_A.append(image_tensor_A)
            image_folder_B.append(image_tensor_B)
            image_label_list.append(image_label)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        max_length = max(tensor.size(1) for tensor in input_batch)

        final_input_list = [torch.cat((torch.zeros((1,max_length - tensor.size(1)), dtype=tensor.dtype,device=tensor.get_device()), tensor),dim=1) for tensor in input_batch]
        final_input_tensors=torch.cat(final_input_list,dim=0)
        #image_tensor_batch = image_processor.preprocess(image_folder,crop_size ={'height': 504, 'width': 504},size = {'shortest_edge': 504}, return_tensors='pt')['pixel_values']
        image_tensor_batch = {'pre': torch.stack(image_folder_A).half().cuda(),
                              'post': torch.stack(image_folder_B).half().cuda(),
                              'targets': torch.stack(image_label_list).cuda()}

        with torch.inference_mode():
            output_ids = model.generate( final_input_tensors, images=image_tensor_batch, do_sample=False , temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=256,length_penalty=2.0, use_cache=True)

        input_token_len = final_input_tensors.shape[1]
        n_diff_input_output = (final_input_tensors != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        for k in range(0,len(final_input_list)):
            output = outputs[k].strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()

            ans_id = shortuuid.uuid()
            
            ans_file.write(json.dumps({
                                    "question_id": questions[count]["question"],
                                    "image_id": questions[count]["img_id"],
                                    "answer": output,
                                    }) + "\n")
            count=count+1
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./cdchat/cdchat/checkpoints/cdchat_lora")
    parser.add_argument("--model-base", type=str, default='./cdchat/llava-v1.5-7b')
    parser.add_argument("--mm-projector-path", type=str, default='./cdchat/cdchat/checkpoints/pretrain_mm_projector/mm_projector.bin')
    parser.add_argument("--image-folder", type=str, default="./dataset/cd_dataset/")
    parser.add_argument("--question-file", type=str, default="./dataset/LEVIR-CD-256/questions_levir_test.json")
    parser.add_argument("--answers-file", type=str, default="./answer_levir.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size",type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
