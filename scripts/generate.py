import torch

def sample_next_token(
    logits:torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
)-> torch.Tensor:

    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")

    scaled_logits = logits/temperature

    prob = torch.softmax(
        scaled_logits,
        dim = -1
    )

    sorted_prob, sorted_indices = torch.sort(
        prob,
        dim = -1,
        descending = True,
    )

    cum_prob = torch.cumsum(
        sorted_prob,
        dim = -1,
    )

    pre_cum_prob = cum_prob - sorted_prob
    tokens_keep = pre_cum_prob < top_p

    sorted_prob = sorted_prob.masked_fill(
        not tokens_keep,
        0.0
    )

    sorted_prob = sorted_prob / sorted_prob.sum(
        dim = -1,
        keepdim = True,
    )

    sample = torch.multimomial(
        sorted_prob,
        num_samples = 1,
    )

    sample_next_token = torch.gather(
        sorted_indices,
        dim = -1,
        index = sample,
    )

    return sample_next_token


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    prompt_token_ids: list[int],
    eos_token_id: int,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | torch.device = "cuda",
) -> list[int]:

    if len(prompt_token_ids) == 0:
        raise ValueError("The prompt must contain at least one token")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    device = torch.device(device)

    was_training = model.training
    model.eval()

    for _ in range(max_new_tokens):
        model_input = generated[:, -context_length:]

        logits = model(model_input)
        next_token_logits = logits[:, -1, :]

        next_token = sample_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_p=top_p,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

        if next_token.item() == eos_token_id:
            break

    if was_training:
        model.train()

    return generated.squeeze(0).tolist()