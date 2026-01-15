import Model_Blocks.Transformer as tfm
import torch
from Data_handler import untokenize

def translate_sentence(model, vectorizer_input, vectorizer_output, sentence,max_len_gen,max_len_input,device):
    model.eval()
    # using vec to tokenize
    tokens = vectorizer_input.build_analyzer()(sentence)  # No wrapping
    print(f"Tokens: {tokens}")  # Debug
    print(f"Token indices: {[vectorizer_input.vocabulary_.get(token) for token in tokens]}")  # Debug
    
    input_indices = [vectorizer_input.vocabulary_['ddd']] + [vectorizer_input.vocabulary_.get(token, vectorizer_input.vocabulary_['PPPading']) for token in tokens] + [vectorizer_input.vocabulary_['fff']] + [vectorizer_input.vocabulary_['PPPading']] * (max_len_input - len(tokens) - 2)
    print(f"Input indices: {input_indices}")  # Debug
    # traduction of the input indices through reverse vocab (for debugging)
    inv_vocab_input = {v: k for k, v in vectorizer_input.vocabulary_.items()}
    reconstructed_sentence = ' '.join([inv_vocab_input[idx] for idx in input_indices if idx in inv_vocab_input])
    print(f"Reconstructed sentence from indices: {reconstructed_sentence}")  # Debug
    if len(input_indices) > max_len_input:
        input_indices = input_indices[:max_len_input]
    input_tensor = torch.LongTensor(input_indices).unsqueeze(0).to(device)  
    mask = (input_tensor == vectorizer_input.vocabulary_['PPPading']).unsqueeze(1)
    mask = mask.float().masked_fill(mask == 1, float('-inf'))

    y = torch.LongTensor([vectorizer_output.vocabulary_['ddd']]).unsqueeze(0).to(device)    
    for _ in range(max_len_gen - 1):
        with torch.no_grad():
            model_output =  model(input_tensor, y, mask_input=mask)
        next_token = model_output[:, -1, :].argmax(dim=-1).unsqueeze(1)
        y = torch.cat([y, next_token], dim=1)
        if next_token.item() == vectorizer_output.vocabulary_['fff']:
            break

    output_indices = y.squeeze(0).tolist()
    output_sentence = untokenize(torch.tensor([output_indices]), vectorizer_output)[0]
    print("Input Sentence: ", sentence)
    print("Translated Sentence: ", output_sentence)
    return output_sentence

