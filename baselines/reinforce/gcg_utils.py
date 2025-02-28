import torch


def sample_control(control_toks, grad, search_width, topk=256, n_replace=1, temp=1, not_allowed_tokens=None,
                   strategy='gradient'):
    control_toks = control_toks.to(grad.device)
    original_control_toks = control_toks.repeat(search_width, 1)
    n_replace = int(n_replace)

    if not_allowed_tokens is not None:
        grad = grad.clone()
        if strategy == 'gradient':  # Was faster in some experiments
            grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1
        else:
            grad[:, not_allowed_tokens.to(grad.device)] = float('Inf')

    if strategy == 'gradient':
        top_indices = (-grad).topk(topk, dim=1).indices

        if n_replace > 1:  # adapted
            # This is not necessarily unique for each candidate. So it is up to n_replace substitutions
            new_token_pos = torch.randint(0, len(control_toks), (search_width, n_replace), device=grad.device)
            new_token_val = torch.gather(
                top_indices[new_token_pos], 2,
                torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)).squeeze(-1)
            control_tok_idx = torch.arange(0, search_width, device=grad.device)[:, None].repeat(1, n_replace)
            original_control_toks[control_tok_idx, new_token_pos] = new_token_val
            new_control_toks = original_control_toks
        else:  # original
            new_token_pos = torch.arange(
                0,
                len(control_toks),
                len(control_toks) / search_width,
                device=grad.device
            ).to(torch.int64)
            new_token_val = torch.gather(
                top_indices[new_token_pos], 1, torch.randint(0, topk, (search_width, 1), device=grad.device))
            new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    elif strategy == 'random':
        assert n_replace == 1, f"Random sampling only supports n_replace=1 but got {n_replace}"

        new_token_pos = torch.arange(
            0,
            len(control_toks),
            len(control_toks) / search_width,
            device=grad.device
        ).to(torch.int64)

        # Sample unformly from all tokens with finite gradient - increadibly slow on a gpu
        top_indices = torch.where(grad.isfinite().any(0).cpu())[0].to(grad.device)

        new_token_val = torch.gather(
            top_indices, 0, torch.randint(0, top_indices.shape[0], (search_width,), device=grad.device))
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val.unsqueeze(-1))
    else:
        raise NotImplementedError(f"Sample strategy {strategy} not implemented")

    return new_control_toks


def interleave_lists(shorter, longer):
    """Interleave two lists where len(longer) is multiple of len(shorter)."""
    ratio = len(longer) // len(shorter)
    result = []
    for i in range(len(shorter)):
        # Add one item from shorter list
        result.append(shorter[i])
        # Add ratio number of items from longer list
        result.extend(longer[i * ratio:(i + 1) * ratio])
    return result


def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    if "Baichuan2" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(101, 1000)]

    return torch.tensor(ascii_toks, device=device)


def x_bounded_sigmoid(x: torch.Tensor | float, k: int = 2) -> torch.Tensor | float:
    "Sigmoidal / S-curve mapping from [0,1] -> [0,1]"
    return 1 / (1 + (1 / x - 1)**k)
