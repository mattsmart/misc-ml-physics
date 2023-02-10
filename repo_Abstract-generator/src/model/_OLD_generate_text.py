import __init__  # For now, needed for all the relative imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import log_softmax
import torch
from torchtext.datasets import WikiText2
# need all class definitions for un-pickle
from src.model.transformer_torch import TransformerModel, PositionalEncoding, load_model
from src.settings import DIR_MODELS, BPTT


def tokenize_some_text(tokenizer, vocab, text='The dog ran across the'):
    """
    Current vanilla pytorch tokenizer
        1) Split by spaces into word tokens
        2) Use vocab to creat integer rep of each token
    """
    tokenized_text = tokenizer(text)
    tokenized_text_ints = torch.tensor([vocab[token] for token in tokenized_text], dtype=torch.long)
    return tokenized_text, tokenized_text_ints


def gen_some_text(model, tokenizer, vocab, device,
                  text_prompt='The dog ran across the',
                  tokens_to_gen=10,
                  vis=False,
                  verbose=True,
                  sidestep_unk=False,
                  decode_style='greedy',
                  decode_seed=0,
                  decode_beta=1.0,
                  decode_sample_topp_threshold=0.70):
    """
    dummy_token: if text_prompt is < BPTT tokens, need to add dummy tokens and mask them
    1) tokenize the text prompt
    If vis: plot distribution over vocab for next word, given the past BPTT
    If verbose: get ready for too many prints
    If sidestep_unk, guess the next most probable word whenever <unk> would have been guessed
    Hyperparameters:
    - decode_sample_topp_threshold: try 0.5 to 0.9 for
    - beta 1.0 works OK, above 2.0 seems close to greedy. note beta=1.0 implicitly used in training
    See also: https://huggingface.co/blog/how-to-generate
    """
    # decoder hyperparameters
    np.random.seed(decode_seed)
    # total_text_string = text_prompt  # this will be extended by tokens_to_gen
    total_text_string = ''

    def process_prompt(dummy_token=0):
        # Two cases:
        # - if less than BPTT (context length), need to add dummy tokens
        # - if longer than BPTT (context length), truncate to BPTT
        text_split, tokenized_text = tokenize_some_text(tokenizer, vocab,
                                                        text=text_prompt)
        nn = tokenized_text.shape[0]
        if verbose:
            print(tokenized_text)
            print(nn)
        if nn > BPTT:  # take last BPTT elements
            input_slice = tokenized_text[nn - BPTT:]
            src_mask = model.generate_square_subsequent_mask(BPTT).to(device)
        else:
            input_slice = tokenized_text[0:nn]
            src_mask = model.generate_square_subsequent_mask(nn).to(device)
        src = torch.zeros((min(nn, BPTT), 1), dtype=torch.long).to(device)
        src[0:nn, 0] = input_slice
        return text_split, src, src_mask
    def decode(model_out, nn, style='greedy'):
        """
        Sampling notes:
        - working in log space to improve numeric stability
        - use "Gumbel-max trick"
          - https://stats.stackexchange.com/questions/64081/how-do-i-sample-from-a-discrete-categorical-distribution-in-log-space
          - https://en.wikipedia.org/wiki/Categorical_distribution#Sampling_via_the_Gumbel_distribution
        TODO - verify that this trick works the same as the classic way, for stability we need to compare with high noise (low beta) -- how to make the random samples identical?
        """
        assert style in ['greedy', 'sample_full', 'sample_topp']
        if style == 'greedy':
            guessed_int = torch.argmax(model_out[nn - 1, 0])  # care batch dimension, nn vs BPTT
        elif style == 'sample_full':
            """
            next_word_weights = out[nn-1, 0].detach().numpy()
            next_word_weights_exp = np.exp(DECODE_SAMPLE_BETA * next_word_weights)
            print('exp check')
            print(next_word_weights_exp)
            next_word_probs = next_word_weights_exp / np.sum(next_word_weights_exp)
            print("LIMITS::::::::::::::::", np.min(next_word_probs), np.max(next_word_probs))
            print("LIMITS::::::::::::::::", np.min(next_word_weights_exp), np.max(next_word_weights_exp))
            print("LIMITS::::::::::::::::", np.sum(next_word_weights_exp))
            unirand = np.random.rand()
            distribution_cumsum = np.cumsum(next_word_probs)
            guessed_int = np.searchsorted(distribution_cumsum, unirand)
            print('BBBBBBBBBBBBBBBBBBBBBBBBBBB', type(unirand), unirand)
            print(distribution_cumsum)
            print(guessed_int)
            print(distribution_cumsum[guessed_int-1], distribution_cumsum[guessed_int], distribution_cumsum[guessed_int+1])
            print()"""
            # numerically stable approach: gumbel max-trick sampling
            next_word_weights = out[nn - 1, 0].cpu().detach().numpy()
            ncategories = next_word_weights.shape[0]
            next_word_weights_scaled = decode_beta * next_word_weights
            if verbose:
                print("ncategories", ncategories)
            uvec = np.random.rand(ncategories)
            gvec = -np.log(-np.log(uvec))
            guessed_int = np.argmax(gvec + next_word_weights_scaled)
        else:
            assert style == 'sample_topp'
            # TODO implement gumbel max trick here too
            print('TODO implement gumbel max trick for decode_style == sample_topp')
            next_word_weights = out[nn - 1, 0].cpu().detach().numpy()
            next_word_weights_exp = np.exp(decode_beta * next_word_weights)
            next_word_probs = next_word_weights_exp / np.sum(next_word_weights_exp)
            # 1) identify top p words such their cumulative probability passes threshold
            distribution_sorted_indices = np.argsort(next_word_probs)[::-1]
            next_word_probs_descsort = next_word_probs[distribution_sorted_indices]
            next_word_probs_descsort_cumsum = np.cumsum(next_word_probs_descsort)
            threshold_index = np.searchsorted(next_word_probs_descsort_cumsum,
                                              decode_sample_topp_threshold)
            # 2) sample from these top p words
            topp_indices = distribution_sorted_indices[:threshold_index + 1]
            if verbose:
                print(distribution_sorted_indices)
                print(next_word_probs_descsort_cumsum)
                print("topp_indices", len(topp_indices), topp_indices)
            topp_probs = next_word_probs_descsort[:threshold_index + 1]
            topp_reweighted_probs = topp_probs / np.sum(topp_probs)
            topp_reweighted_cumsum = np.cumsum(topp_reweighted_probs)
            unirand = np.random.rand()
            topp_choice = np.searchsorted(topp_reweighted_cumsum, unirand)
            guessed_int = topp_indices[topp_choice]
        return guessed_int
    # 1) tokenize the text prompt and prepare associated src_mask for model.forward()
    prompt_split, src, src_mask = process_prompt()  # src should be in form ntokens x nbatches
    nn = src_mask.shape[0]
    src.reshape((nn, 1))
    if verbose:
        print(src)
        print(src_mask)
    # 2)
    model.eval()
    for idx in range(tokens_to_gen):
        running_context_string = ' '.join([vocab.itos[src[k]] for k in range(src.shape[0])])
        # TESTING DIFFERENT src_mask (all zero)
        use_diff_mask = False
        if use_diff_mask:
            print('Warning: testing maskless generation')
            # A:
            # src_mask = torch.zeros(nn, nn)
            # B:
            src_mask = torch.zeros(nn, nn) + float('-inf')
            src_mask[:, 0] = 0
            src_mask[-1, :] = 0
            # C:
            # src_mask = torch.rand(nn, nn)
            # print(src_mask)
            src_mask.to(device)
        out = model.forward(src, src_mask)
        if sidestep_unk:
            unk_index = vocab.unk_index
            # print(out.size(), unk_index, vocan.itos[unk_index])
            MODEL_OUTPUT_MINIMAL = -1e9
            out[:, :, unk_index] = MODEL_OUTPUT_MINIMAL
        if verbose:
            print(out.shape)
        if vis:
            next_word_weights = out[nn - 1, 0].detach().numpy()
            next_word_weights_exp = np.exp(next_word_weights)
            next_word_probs = next_word_weights_exp / np.sum(next_word_weights_exp)
            plt.plot(next_word_weights)
            plt.title('next_word_weights: iteration %d' % idx)
            plt.xlabel('vocab index')
            plt.ylabel('weight')
            # plt.xlim((-0.5,100.5))
            plt.show()
            plt.plot(next_word_probs)
            plt.title('next_word_probs: iteration %d' % idx)
            plt.xlabel('vocab index')
            plt.ylabel('probability')
            # plt.xlim((-0.5,100.5))
            plt.show()
            kk = 20
            top_word_indices = np.argsort(next_word_probs)[::-1]
            top_word_probs = next_word_probs[top_word_indices]
            x = list(range(kk))
            plt.title('next_word_probs: iteration %d' % idx)
            plt.suptitle(running_context_string + ' ???', fontsize=8, wrap=True)
            plt.xticks(x, [vocab.itos[k] for k in top_word_indices[0:kk]], rotation=60)
            plt.plot(x, top_word_probs[0:kk])
            plt.ylabel('probability')
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.show()
        next_guess_int = decode(out, nn, style=decode_style)
        next_guess_string = vocab.itos[next_guess_int]
        if verbose:
            print('next_guess_int, next_guess_string:', next_guess_int, next_guess_string)
        # update total_text_string by adding best guess
        total_text_string += ' %s' % next_guess_string
        # update src and src mask for next pass of model.forward()
        if nn < BPTT:
            # extend and shift the running window of input data
            src_updated = torch.zeros((nn + 1, 1), dtype=torch.long)  # extend src to nn
            src_updated[0:nn, 0] = src[:, 0]
            src_updated[nn, 0] = next_guess_int
            src = src_updated.to(device)
            # increment mask dimension
            src_mask = model.generate_square_subsequent_mask(nn + 1).to(device)
            nn += 1
        else:
            src_orig = src.clone()
            src[0:BPTT - 1, 0] = src_orig[1:, 0]
            src[-1, 0] = next_guess_int
    return total_text_string


if __name__ == '__main__':
    # device settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # specify path to model or model_weights
    fname = 'bestval_model.pth'  # 'end_model.pth' or 'bestval_model.pth'
    adam_dir = 'modelB_epoch50_batch128_adamW1e-4_drop0.1_bptt35'
    sgd_dir = 'modelB_epoch150_batch64_sgdLRstep0.5_thirdCycle_drop0.3_bptt35'
    model_adam = DIR_MODELS + os.sep + adam_dir + os.sep + fname
    model_sgd = DIR_MODELS + os.sep + sgd_dir + os.sep + fname
    # model_path = model_adam
    model_path = model_sgd

    # TODO reimplement with new tokenizers -- see latest training notebook
    """
    from src.model.model_utils import gen_tokenizer_and_vocab, data_process, batchify
    from src.model._OLD_model_utils import gen_tokenizer_and_vocab, data_process, batchify

    # load dataset, tokenizer, vocab
    tokenizer, vocab = gen_tokenizer_and_vocab()
    train_iter, val_iter, test_iter = WikiText2()
    # load method A
    model_A = load_model(model_path, device, as_pickle=True, vocab=None)
    # inspect model
    print('model_A info...\n', model_A)
    # Text generation example
    # prompt = 'Text generation is easier than you think , however'
    prompt = 'Text generation is easier than you think. More challenging , however , ' \
             'is training the underlying model to generate believable text. In fact,'
    # prompt = 'The dog ran across the'
    ngen = 120
    decode_style = 'sample_full'  # sample_topp, sample_full, or greedy
    print("Text prompt:\n", prompt)
    print("Decode style:", decode_style)
    print("Number of tokens to generate:", ngen)
    for decode_seed in range(0, 4):
        generated_text = gen_some_text(
            model_A, tokenizer, vocab, device,
            text_prompt=prompt,
            tokens_to_gen=ngen,
            decode_style=decode_style,
            decode_seed=decode_seed,
            decode_beta=2.0,
            sidestep_unk=True,
            vis=False,
            verbose=False)
        print("(seed=%d) Generated_text:\n%s" % (decode_seed, generated_text))
    """
