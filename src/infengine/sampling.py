import torch


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_p >= 1.0:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    cutoff = cum_probs > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_logits[cutoff] = -float("inf")
    probs = torch.softmax(sorted_logits, dim=-1)
    samp = torch.multinomial(probs, 1)
    return torch.gather(sorted_idx, -1, samp)